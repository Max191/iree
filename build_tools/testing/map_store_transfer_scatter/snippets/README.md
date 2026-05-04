# `map_store` pre-vectorization snippets

This directory is an indexed corpus of real IR snippets that contain
`iree_linalg_ext.map_store` before generic vectorization or the GPU vector
distribution/decomposition steps that currently lower it. The snippets are
inputs for designing and validating direct `map_store` to
`iree_vector_ext.transfer_scatter` vectorization.

All regeneration commands below intentionally include `--mlir-disable-threading`
so pass dumps are deterministic.

## Focused snippets

| File | Source | Why it matters |
|------|--------|----------------|
| `focused/generic_vectorization_map_store_pre.mlir` | `compiler/src/iree/compiler/Codegen/Common/test/generic_vectorization_map_store.mlir` | Minimal pre-generic-vectorization tensor `map_store` cases, including static, dynamic, sub-byte, stride, repeated-index, and mask-dependent variants. |
| `focused/e2e_linalg_ext_map_store_pre.mlir` | `tests/e2e/linalg_ext_ops/map_store.mlir` | E2E semantic cases for copy-like, collapse-like, expand-like, and masked slice-like `map_store`. |

Keep the E2E semantic copy byte-for-byte identical to the source test. When
either file changes, refresh with the `cp` command below and verify with
`diff -q`.

Regenerate the focused compiler dump:

```bash
DUMP_ROOT=/tmp/map_store_snippet_dumps/focused
mkdir -p "$DUMP_ROOT"

TMPDIR=/tmp build_tools/bin/iree-bazel-run //tools:iree-opt -- \
  --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=false vectorize-map-store=true}))" \
  --split-input-file \
  --mlir-disable-threading \
  --mlir-print-ir-after-all \
  compiler/src/iree/compiler/Codegen/Common/test/generic_vectorization_map_store.mlir \
  2> "$DUMP_ROOT/generic_vectorization.after_all.mlir"
```

Regenerate the E2E compile tree:

```bash
DUMP_ROOT=/tmp/map_store_snippet_dumps/e2e
mkdir -p "$DUMP_ROOT/tree"

TMPDIR=/tmp build_tools/bin/iree-bazel-run //tools:iree-compile -- \
  tests/e2e/linalg_ext_ops/map_store.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=generic \
  --mlir-disable-threading \
  --mlir-print-ir-after-all \
  --mlir-print-ir-tree-dir="$DUMP_ROOT/tree" \
  -o "$DUMP_ROOT/map_store.vmfb" \
  2> "$DUMP_ROOT/compile.after_all.mlir"
```

Expected pre-vectorization dump paths:

- `$DUMP_ROOT/generic_vectorization.after_all.mlir`
- `$DUMP_ROOT/tree/**generic-vectorization*` for the first pass boundary that
  must not be crossed when extracting pre-vectorization snippets
- `$DUMP_ROOT/compile.after_all.mlir` for a headered fallback log

Regenerate the checked-in E2E semantic corpus copy:

```bash
cp tests/e2e/linalg_ext_ops/map_store.mlir \
  build_tools/testing/map_store_transfer_scatter/snippets/focused/e2e_linalg_ext_map_store_pre.mlir
```

## GPU dispatch snippets

| File | Source | Why it matters |
|------|--------|----------------|
| `gpu/rocdl_matmul_map_store_pre_vector_distribute.mlir` | `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_vector_distribute_gfx942.mlir` | Full ROCm dispatch function with matmul output packed through `affine.delinearize_index` before vector distribution. |
| `gpu/rocdl_tile_and_fuse_data_tiling_map_store_pre.mlir` | `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_tile_and_fuse.mlir` | Data-tiling map-store dispatch with two-dimensional delinearization into a tiled output layout. |

Regenerate GPU tree dumps:

```bash
DUMP_ROOT=/tmp/map_store_snippet_dumps/gpu
mkdir -p "$DUMP_ROOT/vector_distribute" "$DUMP_ROOT/tile_and_fuse"

TMPDIR=/tmp build_tools/bin/iree-bazel-run //tools:iree-opt -- \
  --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-vector-distribute))))" \
  --split-input-file \
  --mlir-disable-threading \
  --mlir-print-ir-after-all \
  --mlir-print-ir-tree-dir="$DUMP_ROOT/vector_distribute" \
  compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_vector_distribute_gfx942.mlir \
  2> "$DUMP_ROOT/vector_distribute.after_all.mlir"

TMPDIR=/tmp build_tools/bin/iree-bazel-run //tools:iree-opt -- \
  --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-tile-and-fuse))))" \
  --split-input-file \
  --mlir-disable-threading \
  --mlir-print-ir-after-all \
  --mlir-print-ir-tree-dir="$DUMP_ROOT/tile_and_fuse" \
  compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_tile_and_fuse.mlir \
  2> "$DUMP_ROOT/tile_and_fuse.after_all.mlir"
```

Expected pre-vectorization dump paths:

- `$DUMP_ROOT/vector_distribute/**generic-vectorization*` or
  `$DUMP_ROOT/vector_distribute.after_all.mlir`
- `$DUMP_ROOT/tile_and_fuse/**generic-vectorization*` or
  `$DUMP_ROOT/tile_and_fuse.after_all.mlir`

## BOO convolution snippets

`boo/README.md` is the extraction index for BOO driver convolution cases. It
starts without checked-in IR because BOO requires a configured
ROCm/PyTorch/Turbine environment; add only real dump-derived snippets there.

## Validation

When Bazel and submodules are configured, parse the compiler-IR and GPU dump
snippets:

```bash
find build_tools/testing/map_store_transfer_scatter/snippets \
  \( -path '*/focused/e2e_linalg_ext_map_store_pre.mlir' -prune \) -o \
  -name '*.mlir' -print | LC_ALL=C sort | while read -r snippet; do
  TMPDIR=/tmp build_tools/bin/iree-bazel-run //tools:iree-opt -- \
    --split-input-file --verify-diagnostics "$snippet" >/dev/null
done
```

The E2E-shaped semantic copy is pruned from the parse-only loop because it uses
the E2E `util` and `check` dialect harness. Validate it through the source test
or compile path:

```bash
TMPDIR=/tmp build_tools/bin/iree-bazel-test \
  //tests/e2e/linalg_ext_ops:check_llvm-cpu_local-task_map_store.mlir
```

When the local worktree is not build-configured, run the structural check:

```bash
python3 - <<'PY'
from pathlib import Path
root = Path("build_tools/testing/map_store_transfer_scatter/snippets")
files = sorted(root.glob("**/*.mlir"))
missing = [str(path) for path in files if "iree_linalg_ext.map_store" not in path.read_text()]
assert files, "no snippet files found"
assert not missing, "missing map_store in: " + ", ".join(missing)
print(f"checked {len(files)} snippet files")
PY
```
