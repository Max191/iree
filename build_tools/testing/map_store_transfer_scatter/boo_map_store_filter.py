#!/usr/bin/env python3
"""Filters BOO convolution commands by `iree_linalg_ext.map_store` dumps."""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import traceback
from pathlib import Path


DEFAULT_STOP_PASS_REGEX = "generic-vectorization"
MAP_STORE_OP_PATTERN = re.compile(
    r'^\s*(?:%[\w\d_$.#-]+\s*=\s*)?"?iree_linalg_ext\.map_store"?(?=\s|\(|$)'
)
IR_DUMP_HEADER_PATTERN = re.compile(
    r"^// -+// IR Dump (?P<when>Before|After) (?P<label>.*?)(?://-+ //)?\s*$"
)


def read_commands(commands_file: Path) -> list[str]:
    commands = []
    for line in commands_file.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            commands.append(stripped)
    return commands


def write_commands(commands_file: Path, commands: list[str]) -> None:
    commands_file.parent.mkdir(parents=True, exist_ok=True)
    commands_file.write_text("".join(f"{command}\n" for command in commands))


def get_site_packages(venv: Path) -> str:
    python = venv / "bin" / "python"
    result = subprocess.run(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def build_boo_env(iree_build: Path, venv: Path, cache_dir: Path) -> dict[str, str]:
    site_packages = Path(get_site_packages(venv))
    rocm_library_dir = os.environ.get("ROCM_LIBRARY_DIR")
    rocm_library_dirs = sorted(site_packages.glob("_rocm_sdk_libraries_*/lib"))
    if rocm_library_dir:
        selected_rocm_library_dir = rocm_library_dir
    elif not rocm_library_dirs:
        raise RuntimeError(f"no ROCm library package found in {site_packages}")
    else:
        if len(rocm_library_dirs) > 1:
            print(
                "warning: multiple ROCm library dirs found, using "
                f"{rocm_library_dirs[0]}",
                file=sys.stderr,
            )
        selected_rocm_library_dir = str(rocm_library_dirs[0])
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{iree_build / 'tools'}:{venv / 'bin'}:{environment['PATH']}",
            "PYTHONPATH": (
                f"{iree_build / 'compiler' / 'bindings' / 'python'}:"
                f"{iree_build / 'runtime' / 'bindings' / 'python'}"
            ),
            "LD_LIBRARY_PATH": (
                f"{site_packages / '_rocm_sdk_core' / 'lib'}:"
                f"{selected_rocm_library_dir}"
            ),
            "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=4096",
            "BOO_USE_BACKWARD_KERNELS": "1",
            "BOO_CACHE_ON": "1",
            "BOO_CACHE_DIR": str(cache_dir),
        }
    )
    return environment


def find_compile_command_file(cache_dir: Path) -> Path:
    compile_commands = sorted(cache_dir.glob("**/compile_command_*.txt"))
    if not compile_commands:
        raise FileNotFoundError(f"no compile_command_*.txt found under {cache_dir}")
    if len(compile_commands) > 1:
        matches = "\n".join(str(path) for path in compile_commands)
        raise RuntimeError(
            f"multiple compile_command_*.txt files found under {cache_dir}:\n"
            f"{matches}"
        )
    return compile_commands[0]


def read_compile_command(compile_command_file: Path) -> list[str]:
    text = compile_command_file.read_text().strip()
    if not text:
        raise ValueError(f"{compile_command_file} is empty")
    return shlex.split(text)


def with_debug_flags(command: list[str], dump_dir: Path) -> list[str]:
    filtered = [
        argument
        for argument in command
        if not argument.startswith("--mlir-print-ir-tree-dir=")
        and argument not in {
            "--mlir-disable-threading",
            "--mlir-print-ir-after-all",
        }
    ]
    debug_flags = [
        "--mlir-disable-threading",
        "--mlir-print-ir-after-all",
        f"--mlir-print-ir-tree-dir={dump_dir}",
    ]
    if not filtered:
        return debug_flags
    return [filtered[0], *debug_flags, *filtered[1:]]


def with_resolved_compile_tool(command: list[str], iree_build: Path) -> list[str]:
    """Resolves BOO-captured `iree-compile` spellings against `iree_build`."""
    if not command:
        return command
    command_tool = Path(command[0])
    if command_tool.is_absolute():
        return command
    if command_tool.name != "iree-compile":
        return command
    build_relative_tool = iree_build / command_tool
    tools_basename_tool = iree_build / "tools" / command_tool.name
    for candidate_tool in (build_relative_tool, tools_basename_tool):
        if candidate_tool.exists():
            return [str(candidate_tool), *command[1:]]
    return command


def list_text_files(dump_path: Path) -> list[Path]:
    if dump_path.is_file():
        return [dump_path]
    suffixes = {".mlir", ".txt", ".log"}
    return sorted(
        path
        for path in dump_path.rglob("*")
        if path.is_file() and path.suffix in suffixes
    )


def scan_text(
    text: str,
    stop_before_pass_regex: str,
) -> tuple[bool, list[str]]:
    stop_regex = re.compile(stop_before_pass_regex) if stop_before_pass_regex else None
    in_pre_stop_region = True
    current_header = "<before first IR dump header>"
    saw_header = False
    matching_headers = []

    for line in text.splitlines():
        header_match = IR_DUMP_HEADER_PATTERN.match(line)
        if header_match:
            saw_header = True
            current_header = line.strip()
            is_after_dump = header_match.group("when") == "After"
            if stop_regex and is_after_dump and stop_regex.search(current_header):
                in_pre_stop_region = False
            continue
        if in_pre_stop_region and MAP_STORE_OP_PATTERN.search(line):
            if current_header not in matching_headers:
                matching_headers.append(current_header)

    if not saw_header and matching_headers == ["<before first IR dump header>"]:
        matching_headers = ["<whole file>"]
    return bool(matching_headers), matching_headers


def scan_dumps(
    dump_path: Path,
    stop_before_pass_regex: str,
    stop_before_file_regex: str,
) -> dict[str, object]:
    files = list_text_files(dump_path)
    matches = []
    stop_file_regex = (
        re.compile(stop_before_file_regex) if stop_before_file_regex else None
    )
    for file_path in files:
        if dump_path.is_dir() and stop_file_regex:
            if stop_file_regex.search(file_path.name):
                break
        text = file_path.read_text(errors="replace")
        has_match, headers = scan_text(text, stop_before_pass_regex)
        if has_match:
            matches.append(
                {
                    "file": str(file_path),
                    "sections": headers,
                }
            )
    return {
        "path": str(dump_path),
        "stop_before_pass_regex": stop_before_pass_regex,
        "stop_before_file_regex": stop_before_file_regex,
        "has_pre_vectorization_map_store": bool(matches),
        "matches": matches,
    }


def run_boo_command(
    command: str,
    iree_build: Path,
    venv: Path,
    case_dir: Path,
    verify_numerics: bool,
    timeout_seconds: int,
) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    command_file = case_dir / "command.txt"
    command_file.write_text(f"{command}\n")
    cache_dir = case_dir / "cache"
    environment = build_boo_env(iree_build, venv, cache_dir)
    boo_command = [
        str(venv / "bin" / "python"),
        "-m",
        "iree.turbine.kernel.boo.driver.driver",
        "--commands-file",
        str(command_file),
    ]
    if verify_numerics:
        boo_command.extend(["--verify-numerics", "--numerics-verbose"])
    subprocess.run(
        boo_command,
        check=True,
        cwd=case_dir,
        env=environment,
        text=True,
        timeout=timeout_seconds if timeout_seconds > 0 else None,
    )
    return find_compile_command_file(cache_dir)


def run_debug_compile(
    compile_command_file: Path,
    iree_build: Path,
    dump_dir: Path,
    timeout_seconds: int,
) -> Path:
    compile_command = read_compile_command(compile_command_file)
    compile_command = with_resolved_compile_tool(compile_command, iree_build)
    debug_command = with_debug_flags(compile_command, dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    stderr_file = dump_dir / "compile.after_all.mlir"
    stdout_file = dump_dir / "compile.stdout"
    with stdout_file.open("w") as stdout_handle:
        with stderr_file.open("w") as stderr_handle:
            subprocess.run(
                debug_command,
                check=True,
                cwd=compile_command_file.parent,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=timeout_seconds if timeout_seconds > 0 else None,
            )
    return stderr_file


def command_filter_convs(args: argparse.Namespace) -> int:
    commands = read_commands(args.commands_file)
    if args.limit > 0:
        commands = commands[: args.limit]

    matches = []
    nonmatches = []
    errors = []
    records = []
    for command_index, command in enumerate(commands):
        case_dir = args.work_dir / f"case_{command_index:04d}"
        print(f"[{command_index + 1}/{len(commands)}] {command}", flush=True)
        try:
            compile_command_file = run_boo_command(
                command,
                args.iree_build,
                args.venv,
                case_dir,
                args.verify_numerics,
                args.timeout_seconds,
            )
            dump_dir = case_dir / "debug_dumps"
            stderr_file = run_debug_compile(
                compile_command_file,
                args.iree_build,
                dump_dir,
                args.timeout_seconds,
            )
            scan_result = scan_dumps(
                dump_dir,
                args.stop_before_pass_regex,
                args.stop_before_file_regex,
            )
            matched = bool(scan_result["has_pre_vectorization_map_store"])
            if matched:
                matches.append(command)
            else:
                nonmatches.append(command)
            records.append(
                {
                    "command": command,
                    "status": "matched" if matched else "nonmatch",
                    "matched": matched,
                    "case_dir": str(case_dir),
                    "compile_command_file": str(compile_command_file),
                    "stderr_dump": str(stderr_file),
                    "scan": scan_result,
                }
            )
        except Exception as error:
            errors.append(command)
            records.append(
                {
                    "command": command,
                    "status": "error",
                    "matched": None,
                    "case_dir": str(case_dir),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            if not args.keep_going:
                raise
            print(f"error: {error}", file=sys.stderr)

    write_commands(args.matches_out, matches)
    write_commands(args.nonmatches_out, nonmatches)
    write_commands(args.errors_out, errors)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(records, indent=2))
    print(f"matched {len(matches)} of {len(commands)} commands")
    print(f"nonmatched {len(nonmatches)} of {len(commands)} commands")
    print(f"errors {len(errors)} of {len(commands)} commands")
    print(f"matches: {args.matches_out}")
    print(f"nonmatches: {args.nonmatches_out}")
    print(f"errors: {args.errors_out}")
    print(f"report: {args.report_out}")
    return 2 if errors else 0


def command_scan_dumps(args: argparse.Namespace) -> int:
    result = scan_dumps(
        args.dump_path,
        args.stop_before_pass_regex,
        args.stop_before_file_regex,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["has_pre_vectorization_map_store"] else 1


def command_augment_compile_command(args: argparse.Namespace) -> int:
    command = read_compile_command(args.compile_command_file)
    debug_command = with_debug_flags(command, args.dump_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(shlex.join(debug_command) + "\n")
    else:
        print(shlex.join(debug_command))
    return 0


def command_select_small(args: argparse.Namespace) -> int:
    commands = read_commands(args.source)
    selected = commands[: args.count]
    write_commands(args.output, selected)
    print(f"wrote {len(selected)} commands to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    scan_parser = subparsers.add_parser(
        "scan-dumps",
        help="scan an after-all log or tree-dir for pre-vectorization map_store",
    )
    scan_parser.add_argument("dump_path", type=Path)
    scan_parser.add_argument(
        "--stop-before-pass-regex",
        default=DEFAULT_STOP_PASS_REGEX,
        help="pass header regex that terminates pre-vectorization scanning",
    )
    scan_parser.add_argument(
        "--stop-before-file-regex",
        default="",
        help=(
            "tree-dir filename regex that stops ordered file scanning; use this "
            "only when dump filenames are known to be pass-order prefixed"
        ),
    )
    scan_parser.set_defaults(func=command_scan_dumps)

    augment_parser = subparsers.add_parser(
        "augment-compile-command",
        help="print a captured iree-compile command with debug dump flags",
    )
    augment_parser.add_argument("compile_command_file", type=Path)
    augment_parser.add_argument("--dump-dir", type=Path, required=True)
    augment_parser.add_argument("--output", type=Path)
    augment_parser.set_defaults(func=command_augment_compile_command)

    select_parser = subparsers.add_parser(
        "select-small",
        help="write the first N commands from a filtered commands file",
    )
    select_parser.add_argument("source", type=Path)
    select_parser.add_argument("output", type=Path)
    select_parser.add_argument("--count", type=int, default=10)
    select_parser.set_defaults(func=command_select_small)

    filter_parser = subparsers.add_parser(
        "filter-convs",
        help="run BOO commands and filter those with pre-vectorization map_store",
    )
    filter_parser.add_argument("--commands-file", type=Path, required=True)
    filter_parser.add_argument("--iree-build", type=Path, required=True)
    filter_parser.add_argument("--venv", type=Path, required=True)
    filter_parser.add_argument("--work-dir", type=Path, required=True)
    filter_parser.add_argument("--matches-out", type=Path, required=True)
    filter_parser.add_argument("--nonmatches-out", type=Path, required=True)
    filter_parser.add_argument(
        "--errors-out",
        type=Path,
        default=Path("boo_map_store_filter_errors.txt"),
    )
    filter_parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("boo_map_store_filter_report.json"),
    )
    filter_parser.add_argument("--limit", type=int, default=0)
    filter_parser.add_argument(
        "--stop-before-pass-regex",
        default=DEFAULT_STOP_PASS_REGEX,
        help="pass header regex that terminates pre-vectorization scanning",
    )
    filter_parser.add_argument(
        "--stop-before-file-regex",
        default=DEFAULT_STOP_PASS_REGEX,
        help=(
            "tree-dir filename regex that stops ordered file scanning; BOO "
            "debug dumps use MLIR pass-order filenames"
        ),
    )
    filter_parser.add_argument("--verify-numerics", action="store_true")
    filter_parser.add_argument("--keep-going", action="store_true")
    filter_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=0,
        help="Optional timeout for each BOO and debug compile subprocess.",
    )
    filter_parser.set_defaults(func=command_filter_convs)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "report_out") and not args.report_out.is_absolute():
        args.report_out = args.work_dir / args.report_out
    if hasattr(args, "errors_out") and not args.errors_out.is_absolute():
        args.errors_out = args.work_dir / args.errors_out
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
