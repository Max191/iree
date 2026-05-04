#!/usr/bin/env python3
"""Tests for BOO map_store dump filtering helpers."""

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


if __name__ == "__main__":
    unittest.main()
