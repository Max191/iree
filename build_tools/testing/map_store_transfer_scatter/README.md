# `map_store` to `transfer_scatter` test matrix

This directory collects the focused test plan for reworking
`iree_linalg_ext.map_store` vectorization to lower directly to
`iree_vector_ext.transfer_scatter`.

The matrix is split by feedback speed:

1. Fast lit transform tests.
2. Targeted e2e `map_store` tests.
3. Full CMake `iree-test-deps` compile coverage.
4. BOO convolution correctness and performance runs.
5. Debug dumps for failure reproduction and snippet extraction.

## Fast lit tests

Run these from the IREE source root. Use the wrapper scripts from
`build_tools/bin` and keep `TMPDIR=/tmp` set for Bazel invocations.

```bash
TMPDIR=/tmp build_tools/bin/iree-bazel-test \
  //compiler/src/iree/compiler/Codegen/Common/test:generic_vectorization_map_store.mlir.test

TMPDIR=/tmp build_tools/bin/iree-bazel-test \
  //compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test:decompose_map_store.mlir.test

TMPDIR=/tmp build_tools/bin/iree-bazel-test \
  //compiler/src/iree/compiler/Codegen/LLVMGPU/test:vector_lowering.mlir.test
```

Coverage:

- `generic_vectorization_map_store.mlir` covers current tensor
  `map_store` vectorization and should change from vector-semantics
  `map_store` checks to tensor-semantics `transfer_scatter` checks.
- `VectorizableOpInterface.cpp` contains the current
  `MapStoreOpVectorizationModel`: it reads the full input tensor with
  `vector.transfer_read` and clones `map_store` with a vector input.
- `decompose_map_store.mlir` covers the current post-bufferization
  vector-semantics `map_store` decomposition to `vector.scatter`,
  `vector.store`, and masked stores.
- `vector_lowering.mlir` covers existing `iree_vector_ext.transfer_scatter`
  lowering, including tensor semantics, masks, contiguous inner dimensions,
  and transposed index vectors.
- `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/test` covers
  `transfer_scatter` parsing, verification, canonicalization, tensor/memref
  forms, masks, all-true/all-false masks, and contiguous write folding.
- `iree_comprehensive_bufferize.mlir` covers tensor-semantics
  `transfer_scatter` bufferization.

Additional GPU pipeline tests to keep in the focused matrix:

```bash
TMPDIR=/tmp build_tools/bin/iree-bazel-test \
  //compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/test:roundtrip.mlir.test \
  //compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/test:canonicalize.mlir.test \
  //compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/test:invalid.mlir.test

TMPDIR=/tmp build_tools/bin/iree-bazel-test \
  //compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL:pipeline_vector_distribute_gfx942.mlir.test \
  //compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL:pipeline_tile_and_fuse.mlir.test
```

Current known gaps:

- No existing test asserts direct vectorization from tensor `map_store` to
  tensor-semantics `iree_vector_ext.transfer_scatter`.
- Existing GPU pipeline checks still prove the legacy
  `map_store -> DecomposeMapStore -> vector.scatter` path.
- Focused generated-`transfer_scatter` coverage is still needed for masks,
  affine linearize/delinearize rank changes, sub-byte eligibility, dynamic
  fallback, and contiguity-preserving indexing maps.

## Targeted e2e tests

Use the existing `tests/e2e/linalg_ext_ops/map_store.mlir` checks first.

```bash
TMPDIR=/tmp build_tools/bin/iree-bazel-test \
  //tests/e2e/linalg_ext_ops:check_vmvx_local-task_map_store.mlir

TMPDIR=/tmp build_tools/bin/iree-bazel-test \
  //tests/e2e/linalg_ext_ops:check_llvm-cpu_local-task_map_store.mlir
```

On ROCm-enabled machines, also run:

```bash
TMPDIR=/tmp build_tools/bin/iree-bazel-test \
  //tests/e2e/linalg_ext_ops:check_rocm_hip_map_store.mlir
```

These tests prove runtime semantics for copy-like, collapse-like,
expand-like, and masked slice-like `map_store` forms.

## Full e2e compile coverage

The project-level gate is the CMake `iree-test-deps` target. From the worktree
root after configuring `iree-build`:

```bash
cmake --build iree-build --target iree-test-deps -- -k 0
```

This target builds the generated files needed by e2e tests. It is intentionally
broader than the focused `map_store` checks and should be run before claiming
the lowering stack is end-to-end compile clean.

## Debug dumps

Use deterministic, single-threaded MLIR dumps for failures:

```bash
DUMP_ROOT=/tmp/map_store_dumps
mkdir -p "$DUMP_ROOT"

TMPDIR=/tmp build_tools/bin/iree-bazel-run //tools:iree-opt -- \
  --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=false vectorize-map-store=true}))" \
  --split-input-file \
  --mlir-disable-threading \
  --mlir-print-ir-after-all \
  compiler/src/iree/compiler/Codegen/Common/test/generic_vectorization_map_store.mlir \
  2> "$DUMP_ROOT/generic_vectorization.after_all.mlir"
```

Prefer `--mlir-print-ir-tree-dir=<dir>` when running full `iree-compile`
pipelines because it avoids one huge stderr log:

```bash
DUMP_ROOT=/tmp/map_store_dumps
mkdir -p "$DUMP_ROOT/tree"

TMPDIR=/tmp build_tools/bin/iree-bazel-run //tools:iree-compile -- \
  tests/e2e/linalg_ext_ops/map_store.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=generic \
  --mlir-disable-threading \
  --mlir-print-ir-after-all \
  --mlir-print-ir-tree-dir="$DUMP_ROOT/tree" \
  -o /tmp/map_store.vmfb \
  2> "$DUMP_ROOT/compile.after_all.mlir"
```

Scan a log or tree directory for pre-vectorization `map_store` occurrences:

```bash
python3 build_tools/testing/map_store_transfer_scatter/boo_map_store_filter.py \
  scan-dumps "$DUMP_ROOT/tree" \
  --stop-before-pass-regex generic-vectorization \
  --stop-before-file-regex generic-vectorization
```

Use `--stop-before-file-regex` only for tree dirs whose filenames are ordered
by pass execution. For a single `--mlir-print-ir-after-all` log, the pass-header
regex is sufficient. `scan-dumps` exits `0` when a match is found and `1` when
no pre-vectorization `map_store` match is found. Headerless directory dumps need
`--stop-before-file-regex`; headerless concatenated logs cannot prove the stop
pass boundary. `scan-dumps` leaves filename stopping off by default; `filter-convs`
defaults it on because BOO debug dumps are generated into pass-ordered tree dirs.

## BOO convolution matrix

The BOO driver is optional until BOO is configured on the machine. It expects a
Python environment containing `iree-turbine`, PyTorch ROCm packages, and IREE
Python bindings from a source build.

Seed commands for quick iteration are in:

```text
build_tools/testing/map_store_transfer_scatter/boo_convs_small_seed.txt
```

The larger source list can be any BOO commands file. If available in the
workspace, the historical filtered IGEMM list is a useful starting point; pass
its path through `$FILTERED_CONVS`.

Filter the large list to only commands whose compiler dumps contain
`iree_linalg_ext.map_store` before generic vectorization:

```bash
IREE_BUILD=/path/to/iree-build
VENV=/path/to/.venv
FILTERED_CONVS=/path/to/filtered_convs.txt
WORKDIR=/tmp/boo_map_store_filter
# Optional when multiple ROCm library wheels are installed:
# ROCM_LIBRARY_DIR=/path/to/site-packages/_rocm_sdk_libraries_*/lib

python3 build_tools/testing/map_store_transfer_scatter/boo_map_store_filter.py \
  filter-convs \
  --commands-file "$FILTERED_CONVS" \
  --iree-build "$IREE_BUILD" \
  --venv "$VENV" \
  --work-dir "$WORKDIR" \
  --matches-out "$WORKDIR/map_store_convs.txt" \
  --nonmatches-out "$WORKDIR/non_map_store_convs.txt" \
  --errors-out "$WORKDIR/errors.txt" \
  --stop-before-file-regex generic-vectorization \
  --timeout-seconds 0 \
  --limit 0 \
  --verify-numerics
```

Write the first 10 filtered matches into a deterministic small commands file:

```bash
python3 build_tools/testing/map_store_transfer_scatter/boo_map_store_filter.py \
  select-small "$WORKDIR/map_store_convs.txt" "$WORKDIR/map_store_convs_small.txt" \
  --count 10
```

## BOO correctness and performance

Use fresh BOO caches when switching builds. The cache key does not encode the
compiler revision. `filter-convs` creates one cache under each case directory
to capture BOO's `compile_command_*.txt`; delete `$WORKDIR` when changing
compiler builds. With `--keep-going`, commands that fail BOO setup or replay
are written to `errors.txt`, omitted from `non_map_store_convs.txt`, and make
the command exit `2`.

Run the helper unit tests after editing dump scanning logic:

```bash
python3 -m unittest discover \
  -s build_tools/testing/map_store_transfer_scatter \
  -p '*_test.py'
```

Correctness on the small set:

```bash
IREE_BUILD=/path/to/iree-build
VENV=/path/to/.venv
WORKDIR=/tmp/boo_map_store_filter
SITE=$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")

PATH="$IREE_BUILD/tools:$VENV/bin:$PATH" \
PYTHONPATH="$IREE_BUILD/compiler/bindings/python:$IREE_BUILD/runtime/bindings/python" \
LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$(ls -d "$SITE"/_rocm_sdk_libraries_*/lib | head -1)" \
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096 \
BOO_USE_BACKWARD_KERNELS=1 \
BOO_CACHE_ON=0 \
"$VENV/bin/python" -m iree.turbine.kernel.boo.driver.driver \
  --commands-file "$WORKDIR/map_store_convs_small.txt" \
  --verify-numerics \
  --numerics-verbose
```

Performance comparison against upstream/main:

```bash
IREE_BUILD=/path/to/iree-build
VENV=/path/to/.venv
WORKDIR=/tmp/boo_map_store_filter
SITE=$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")

PATH="$IREE_BUILD/tools:$VENV/bin:$PATH" \
PYTHONPATH="$IREE_BUILD/compiler/bindings/python:$IREE_BUILD/runtime/bindings/python" \
LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$(ls -d "$SITE"/_rocm_sdk_libraries_*/lib | head -1)" \
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096 \
BOO_USE_BACKWARD_KERNELS=1 \
BOO_CACHE_ON=0 \
"$VENV/bin/python" -m iree.turbine.kernel.boo.driver.driver \
  --commands-file "$WORKDIR/map_store_convs.txt" \
  -t 1 \
  --csv "$WORKDIR/results_branch.csv"
```

Repeat with an upstream/main `IREE_BUILD` and write
`results_upstream_main.csv`. Keep the same command file, GPU, ROCm packages,
iteration count, and cache policy for both runs.
