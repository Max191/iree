#!/usr/bin/env python3
"""Tests for BOO map_store dump filtering helpers."""

import json
import tempfile
import unittest
from pathlib import Path

import boo_map_store_filter as filter_tool


def dump_header(pass_name: str) -> str:
    return f"// -----// IR Dump After {pass_name}"


def dump_header_with_suffix(pass_name: str) -> str:
    return f"// -----// IR Dump After {pass_name} //----- //"


def dump_before_header(pass_name: str) -> str:
    return f"// -----// IR Dump Before {pass_name}"


class ScanTextTest(unittest.TestCase):
    def test_finds_map_store_before_stop_header(self):
        text = "\n".join(
            [
                dump_header("tile-and-fuse"),
                "iree_linalg_ext.map_store",
                dump_header_with_suffix("iree-codegen-generic-vectorization"),
                "func.return",
            ]
        )

        has_match, headers = filter_tool.scan_text(text, "generic-vectorization")

        self.assertTrue(has_match)
        self.assertEqual(headers, [dump_header("tile-and-fuse")])

    def test_ignores_map_store_after_stop_header(self):
        text = "\n".join(
            [
                dump_header("tile-and-fuse"),
                "func.return",
                dump_header("iree-codegen-generic-vectorization"),
                "iree_linalg_ext.map_store",
            ]
        )

        has_match, headers = filter_tool.scan_text(text, "generic-vectorization")

        self.assertFalse(has_match)
        self.assertEqual(headers, [])

    def test_finds_map_store_in_before_stop_header_section(self):
        text = "\n".join(
            [
                dump_before_header("iree-codegen-generic-vectorization"),
                "iree_linalg_ext.map_store",
                dump_header("iree-codegen-generic-vectorization"),
                "func.return",
            ]
        )

        has_match, headers = filter_tool.scan_text(text, "generic-vectorization")

        self.assertTrue(has_match)
        self.assertEqual(
            headers,
            [dump_before_header("iree-codegen-generic-vectorization")],
        )

    def test_finds_whole_file_without_stop_regex(self):
        text = "  %0 = iree_linalg_ext.map_store %input into %output"

        has_match, headers = filter_tool.scan_text(text, "")

        self.assertTrue(has_match)
        self.assertEqual(headers, ["<whole file>"])

    def test_finds_quoted_generic_op_form(self):
        text = '  %0 = "iree_linalg_ext.map_store"(%input, %output)'

        has_match, headers = filter_tool.scan_text(text, "")

        self.assertTrue(has_match)
        self.assertEqual(headers, ["<whole file>"])

    def test_ignores_comment_mentions(self):
        text = "// iree_linalg_ext.map_store appears in a CHECK comment"

        has_match, headers = filter_tool.scan_text(text, "")

        self.assertFalse(has_match)
        self.assertEqual(headers, [])


class ScanDumpsTest(unittest.TestCase):
    def test_scans_single_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dump_file = Path(temp_dir) / "compile.after_all.mlir"
            dump_file.write_text("iree_linalg_ext.map_store\n")

            result = filter_tool.scan_dumps(dump_file, "generic-vectorization", "")

            self.assertTrue(result["has_pre_vectorization_map_store"])

    def test_parent_path_does_not_trigger_stop(self):
        with tempfile.TemporaryDirectory(prefix="generic-vectorization-") as temp_dir:
            dump_dir = Path(temp_dir) / "dumps"
            dump_dir.mkdir()
            (dump_dir / "000-before.mlir").write_text(
                "iree_linalg_ext.map_store\n"
            )

            result = filter_tool.scan_dumps(
                dump_dir,
                "generic-vectorization",
                "generic-vectorization",
            )

            self.assertTrue(result["has_pre_vectorization_map_store"])


class HelperTest(unittest.TestCase):
    def test_read_compile_command_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            command_file = Path(temp_dir) / "compile_command_0.txt"
            command_file.write_text("\n")

            with self.assertRaisesRegex(ValueError, "is empty"):
                filter_tool.read_compile_command(command_file)

    def test_read_commands_ignores_blank_and_comment_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            commands_file = Path(temp_dir) / "commands.txt"
            commands_file.write_text("# comment\n\nconv_a\n  conv_b  \n")

            self.assertEqual(
                filter_tool.read_commands(commands_file),
                ["conv_a", "conv_b"],
            )

    def test_with_debug_flags_replaces_existing_debug_flags(self):
        command = [
            "iree-compile",
            "--mlir-disable-threading",
            "--mlir-print-ir-after-all",
            "--mlir-print-ir-tree-dir=/old",
            "input.mlir",
        ]

        debug_command = filter_tool.with_debug_flags(command, Path("/new"))

        self.assertEqual(
            debug_command,
            [
                "iree-compile",
                "--mlir-disable-threading",
                "--mlir-print-ir-after-all",
                "--mlir-print-ir-tree-dir=/new",
                "input.mlir",
            ],
        )

    def test_with_debug_flags_handles_only_stripped_flags(self):
        command = [
            "--mlir-disable-threading",
            "--mlir-print-ir-after-all",
        ]

        debug_command = filter_tool.with_debug_flags(command, Path("/new"))

        self.assertEqual(
            debug_command,
            [
                "--mlir-disable-threading",
                "--mlir-print-ir-after-all",
                "--mlir-print-ir-tree-dir=/new",
            ],
        )

    def test_with_resolved_compile_tool_resolves_bare_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            iree_build = Path(temp_dir) / "iree-build"
            tools_dir = iree_build / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "iree-compile").write_text("")

            resolved = filter_tool.with_resolved_compile_tool(
                ["iree-compile", "input.mlir"],
                iree_build,
            )

            self.assertEqual(
                resolved,
                [str(tools_dir / "iree-compile"), "input.mlir"],
            )

    def test_with_resolved_compile_tool_keeps_empty_command(self):
        self.assertEqual(
            filter_tool.with_resolved_compile_tool([], Path("/unused")),
            [],
        )

    def test_with_resolved_compile_tool_resolves_build_relative_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            iree_build = Path(temp_dir) / "iree-build"
            tools_dir = iree_build / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "iree-compile").write_text("")

            resolved = filter_tool.with_resolved_compile_tool(
                ["tools/iree-compile", "input.mlir"],
                iree_build,
            )

            self.assertEqual(
                resolved,
                [str(tools_dir / "iree-compile"), "input.mlir"],
            )

    def test_with_resolved_compile_tool_resolves_relative_tool_by_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            iree_build = Path(temp_dir) / "iree-build"
            tools_dir = iree_build / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "iree-compile").write_text("")

            resolved = filter_tool.with_resolved_compile_tool(
                ["./iree-compile", "input.mlir"],
                iree_build,
            )

            self.assertEqual(
                resolved,
                [str(tools_dir / "iree-compile"), "input.mlir"],
            )

    def test_with_resolved_compile_tool_keeps_explicit_tool(self):
        resolved = filter_tool.with_resolved_compile_tool(
            ["/opt/iree/tools/iree-compile", "input.mlir"],
            Path("/unused"),
        )

        self.assertEqual(
            resolved,
            ["/opt/iree/tools/iree-compile", "input.mlir"],
        )

    def test_with_resolved_compile_tool_keeps_missing_bare_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            iree_build = Path(temp_dir) / "iree-build"
            (iree_build / "tools").mkdir(parents=True)

            resolved = filter_tool.with_resolved_compile_tool(
                ["iree-compile", "input.mlir"],
                iree_build,
            )

            self.assertEqual(resolved, ["iree-compile", "input.mlir"])

    def test_with_resolved_compile_tool_keeps_other_tool_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            iree_build = Path(temp_dir) / "iree-build"
            tools_dir = iree_build / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "other-tool").write_text("")

            resolved = filter_tool.with_resolved_compile_tool(
                ["other-tool", "input.mlir"],
                iree_build,
            )

            self.assertEqual(resolved, ["other-tool", "input.mlir"])

    def test_filename_stop_prevents_later_tree_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dump_dir = Path(temp_dir)
            (dump_dir / "000-before.mlir").write_text("func.return\n")
            (dump_dir / "001-generic-vectorization.mlir").write_text(
                "func.return\n"
            )
            (dump_dir / "002-after.mlir").write_text(
                "iree_linalg_ext.map_store\n"
            )

            result = filter_tool.scan_dumps(
                dump_dir,
                "generic-vectorization",
                "generic-vectorization",
            )

            self.assertFalse(result["has_pre_vectorization_map_store"])

    def test_directory_without_filename_stop_scans_all_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dump_dir = Path(temp_dir)
            (dump_dir / "000-before.mlir").write_text("func.return\n")
            (dump_dir / "001-generic-vectorization.mlir").write_text(
                "func.return\n"
            )
            (dump_dir / "002-after.mlir").write_text(
                "iree_linalg_ext.map_store\n"
            )

            result = filter_tool.scan_dumps(
                dump_dir,
                "generic-vectorization",
                "",
            )

            self.assertTrue(result["has_pre_vectorization_map_store"])


class SampleArtifactTest(unittest.TestCase):
    def test_boo_sample_commands_match_report(self):
        sample_dir = Path(__file__).parent
        commands_file = sample_dir / "boo_convs_map_store_sample.txt"
        report_file = sample_dir / "boo_convs_map_store_sample_report.json"
        commands = filter_tool.read_commands(commands_file)
        report = json.loads(report_file.read_text())

        self.assertEqual(
            commands,
            [case["command"] for case in report["matched_cases"]],
        )
        self.assertEqual(len(commands), report["matched"])
        self.assertEqual(report["errors"], 0)

        aggregate_counts = {}
        for case in report["matched_cases"]:
            for op_name, count in case["region_ops"].items():
                aggregate_counts[op_name] = aggregate_counts.get(op_name, 0) + count

        self.assertEqual(aggregate_counts, report["aggregate_region_ops"])


if __name__ == "__main__":
    unittest.main()
