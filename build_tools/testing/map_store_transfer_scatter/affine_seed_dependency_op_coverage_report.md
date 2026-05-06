# Affine seed-dependency coverage for `map_store` regions

Report for bead `bd-va2`: real-world `iree_linalg_ext.map_store` index
expressions before generic vectorization / vector distribution.

## Summary

The checked-in focused and GPU snippets show that the initial affine
seed-dependency interface coverage is enough for simple affine offsets,
constant-stride rejection cases, and collapse-like `affine.linearize_index`
cases. It is not enough for the data-tiling and GPU-packed cases that
delinearize a source map-store block argument into tiled destination
dimensions.

The important blocker is not a missing interface model for a new operation
name. It is the precision and context available for
`affine.delinearize_index`: the current model correctly returns per-seed
unknown coefficients for quotient/remainder results because those expressions
are not globally affine in the source linear index. Real data-tiling cases need
either a guarded no-wrap/range proof for map-store tiles or a later consumer
that treats these delinearized tiled layouts as a structured non-affine case.

`arith.index_cast` / `arith.index_castui` are a secondary coverage gap. In the
representative map-store region bodies inspected below, including the BOO
sample, these casts were not observed. They do appear in nearby data-tiling
workload setup and larger ROCm pipeline tests. If a future pre-analysis dump
has map-store indices or dynamic bases flowing through these casts, unsupported
casts will become full `Unknown`.

## Glossary

- BOO: the Turbine convolution benchmark driver workflow documented in
  `build_tools/testing/map_store_transfer_scatter/snippets/boo/README.md`.
- Bead: the workspace issue-tracking item; this report was written for
  `bd-va2`.
- Pre-analysis IR: IR after pre-vectorization affine cleanup and before the
  map-store contiguity analysis or generic vectorization consumes the region
  index expressions.

## Inputs Inspected

All commands below are run from the IREE source root unless noted.

| Case | Dump or source | Map-store regions | Notes |
| --- | --- | ---: | --- |
| Focused generic vectorization snippets | `build_tools/testing/map_store_transfer_scatter/snippets/focused/generic_vectorization_map_store_pre.mlir` | 8 | Minimal identity, affine offset, stride, repeated-index, and mask-dependent cases. |
| Focused E2E semantic snippets | `build_tools/testing/map_store_transfer_scatter/snippets/focused/e2e_linalg_ext_map_store_pre.mlir` | 4 | Copy, collapse, expand, and slice-like map stores from `tests/e2e/linalg_ext_ops/map_store.mlir`. |
| ROCm matmul packed output | `build_tools/testing/map_store_transfer_scatter/snippets/gpu/rocdl_matmul_map_store_pre_vector_distribute.mlir` | 1 | Full dispatch with nested `affine.delinearize_index` before vector distribution. |
| ROCm tile-and-fuse data tiling | `build_tools/testing/map_store_transfer_scatter/snippets/gpu/rocdl_tile_and_fuse_data_tiling_map_store_pre.mlir` | 1 | Data-tiling map store with two source dimensions split into four destination dimensions. |
| BOO sampled convolution filter run | `build_tools/testing/map_store_transfer_scatter/boo_convs_map_store_sample_report.json` | 9 | First 20 commands from `$FILTERED_CONVS` produced 9 pre-vectorization map-store regions. |
| Data-tiling configuration test | `compiler/src/iree/compiler/Codegen/LLVMGPU/test/gpu_pipeline_data_tiling.mlir` | 1 generated | Reproducible with the command below; write to a scratch path outside the repo. |
| Full generic map-store test source | `compiler/src/iree/compiler/Codegen/Common/test/generic_vectorization_map_store.mlir` | 24 | Broader source test for explicit hints, rejection cases, masks, and sub-byte cases. |
| Analysis lit coverage | `compiler/src/iree/compiler/Dialect/Util/Transforms/test/test_affine_seed_dependency_analysis.mlir` | 4 map-store-shaped additions | Minimal map-store-motivated query cases with FileCheck expectations for the observed lattice states. |

The data-tiling dump can be regenerated with the same wrapper style used by
`build_tools/testing/map_store_transfer_scatter/snippets/README.md`:

```bash
SCRATCH="${SCRATCH:-/tmp/map_store_snippet_dumps/affine_seed_dependency}"
mkdir -p "$SCRATCH"

TMPDIR=/tmp build_tools/bin/iree-bazel-run //tools:iree-opt -- \
  --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline))))" \
  --split-input-file \
  --mlir-disable-threading \
  compiler/src/iree/compiler/Codegen/LLVMGPU/test/gpu_pipeline_data_tiling.mlir \
  > "$SCRATCH/gpu_pipeline_data_tiling_configured.mlir"
```

The checked-in snippet regeneration commands are documented in
`build_tools/testing/map_store_transfer_scatter/snippets/README.md`.

## BOO Status

The repo-local BOO command lists are:

- `build_tools/testing/map_store_transfer_scatter/boo_convs_small_seed.txt`
- `build_tools/testing/map_store_transfer_scatter/boo_convs_map_store_sample.txt`

The repo-local BOO report artifacts are:

- `build_tools/testing/map_store_transfer_scatter/boo_convs_map_store_sample_report.json`

Limited BOO IR coverage is claimed for the 20-command local sample recorded
here. The run used a BOO-capable environment:

- `FILTERED_CONVS=/path/to/filtered_convs.txt`
- `IREE_BUILD=/path/to/iree-build`
- `VENV=/path/to/.venv`
- `WORKDIR=/tmp/bd-va2-boo-map-store-filter-20`

The command was:

```bash
python3 build_tools/testing/map_store_transfer_scatter/boo_map_store_filter.py \
  filter-convs \
  --commands-file "$FILTERED_CONVS" \
  --iree-build "$IREE_BUILD" \
  --venv "$VENV" \
  --work-dir "$WORKDIR/run" \
  --matches-out "$WORKDIR/map_store_convs.txt" \
  --nonmatches-out "$WORKDIR/non_map_store_convs.txt" \
  --errors-out "$WORKDIR/errors.txt" \
  --report-out "$WORKDIR/boo_map_store_filter_report.json" \
  --stop-before-file-regex generic-vectorization \
  --timeout-seconds 0 \
  --limit 20 \
  --keep-going
```

Result: 9 matched, 11 nonmatched, 0 errors. The summarized JSON report and
operation counts are checked in as
`build_tools/testing/map_store_transfer_scatter/boo_convs_map_store_sample_report.json`.
The sample was run for IR coverage only; it did not pass `--verify-numerics`
and did not run the full 550-command source list. A follow-up validation bead
should run the full BOO matrix with numerics enabled and compare performance to
upstream/main.

## Map-Store Region Operation Inventory

Counts below include only operations inside map-store regions, not surrounding
buffer loads, linalg producers, or checks.

| Source | Relevant region ops |
| --- | --- |
| Focused generic snippets | `arith.constant` 8, `affine.apply` 3, `arith.cmpi` 1 |
| Focused E2E snippets | `affine.linearize_index` 1, `affine.delinearize_index` 1, `arith.constant` 1, `arith.cmpi` 1 |
| ROCm matmul packed output snippet | `affine.delinearize_index` 4 |
| ROCm tile-and-fuse data-tiling snippet | `affine.delinearize_index` 2 |
| BOO sampled convolution filter run | `affine.apply` 38, `affine.linearize_index` 9, `affine.delinearize_index` 9, `arith.cmpi` 36, `arith.andi` 27, `iree_linalg_ext.yield` 9 |
| Generated `gpu_pipeline_data_tiling` dump | `affine.delinearize_index` 4 |
| Full generic map-store source test | `arith.constant` 24, `affine.apply` 14, `arith.cmpi` 3, `arith.addi` 1 |

Current affine seed-dependency external models in
`compiler/src/iree/compiler/ExternalInterfaces/UtilExternalModels.cpp` cover
`arith.constant`, `arith.addi`, `arith.subi`, `arith.muli`,
`arith.select`, `affine.apply`, `affine.linearize_index`, and
`affine.delinearize_index`. The only unsupported ops observed directly in
map-store bodies are `arith.cmpi` and `arith.andi`, and both are used for masks
rather than yielded index values in the inspected cases.

## Observed Analysis Behavior

The checked-in lit test
`compiler/src/iree/compiler/Dialect/Util/Transforms/test/test_affine_seed_dependency_analysis.mlir`
captures the representative map-store region patterns and can be run with:

```bash
TMPDIR=/tmp build_tools/bin/iree-bazel-test \
  //compiler/src/iree/compiler/Dialect/Util/Transforms/test:test_affine_seed_dependency_analysis.mlir.test
```

Representative results:

| Pattern | Result |
| --- | --- |
| `%idx1 + 2` via `affine.apply` | `seed0 = 0, seed1 = 1` |
| `affine.linearize_index [%arg2, %arg3] by (4, 4)` | `seed0 = 4, seed1 = 1` |
| `affine.delinearize_index %arg2 into (4, 4)` | `seed0 = ?` for both results |
| Nested GPU delinearize of `%arg0` / `%arg1` | dependent source seed is `?`; unrelated source seed remains `0` |

This is the expected conservative result: useful affine expressions are known,
non-affine quotient/remainder expressions become per-seed unknown, and
independence from unrelated map-store block arguments is preserved.

## Coverage Gaps and Transfer Rules

### Gap A: `affine.delinearize_index`

Priority: required for E2E expand-like map stores and all inspected
data-tiling / GPU-packed real cases.

Examples:

- `focused/e2e_linalg_ext_map_store_pre.mlir`: linear input index expanded
  with `affine.delinearize_index %arg2 into (4, 4)`.
- `gpu/rocdl_tile_and_fuse_data_tiling_map_store_pre.mlir`: `%arg0` and
  `%arg1` split into `(16, 128)`.
- `gpu/rocdl_matmul_map_store_pre_vector_distribute.mlir`: nested splits
  `(2, 128)`, `(16, 16)`, `(4, 8, 4)`, and `(4, 4)`.
- Generated `gpu_pipeline_data_tiling_configured.mlir`: splits
  `(256, 128)`, `(20, 64)`, `(2, 4, 16)`, and `(2, 4, 8)`.

Design note / proposed transfer rule:

1. If the linear index input is independent of all seeds, every delinearized
   result is independent.
2. If the op is part of an exact `linearize_index` / `delinearize_index`
   inverse with matching static bases and the linearization is disjoint,
   propagate the original component dependencies to the matching results.
   Prefer keeping this as canonicalization when possible so the analysis sees
   the simpler values.
3. If the linear index depends on a seed and no range/alignment/no-wrap fact is
   available, do not assign a numeric coefficient to quotient or remainder
   results. Mark each dependent seed coefficient as `?`, while preserving
   known `0` for unrelated seeds.
4. If a future map-store-aware context proves that a vectorized seed varies
   only within one static delinearization tile and cannot cross the relevant
   basis boundary, the innermost varying result may get coefficient `1` for
   that seed and outer results may get coefficient `0` for that seed. Without
   such a proof, a blanket coefficient-`1` rule for remainders is unsound.
5. Dynamic bases should remain per-seed `?` for dependent seeds unless they
   are folded to constants or the analysis is extended to carry symbolic
   coefficients. If all operands and bases are seed-independent, the results
   are independent.

This is partly an interface-precision gap and partly an analysis-context gap:
the current op interface signature does not receive map-store iteration ranges,
vector tile widths, or alignment facts. Adding only a local op model cannot
soundly prove the common data-tiling no-wrap cases.

### Gap B: `arith.index_cast` and `arith.index_castui`

Priority: likely needed for broader generated pipelines, especially when
dynamic workload constants or packed layout dimensions feed map-store index
expressions. Not a blocker for the checked-in map-store region snippets
inspected here.

Examples:

- `compiler/src/iree/compiler/Codegen/LLVMGPU/test/gpu_pipeline_data_tiling.mlir`
  casts a loaded `i32` workload constant to `index` before data-tiling setup.
- `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_vector_distribute_gfx942.mlir`
  contains several nearby `arith.index_cast` / `arith.index_castui` cases.

Design note / proposed transfer rule:

1. If the cast operand is seed-independent, the result is seed-independent.
2. If the cast is index-preserving for the relevant value range, forward the
   operand coefficient map unchanged.
3. If the cast may truncate, wrap, reinterpret signedness in a way that changes
   the mathematical integer value, or otherwise lacks a no-overflow proof,
   mark dependent seed coefficients as `?` rather than full `Unknown`.
4. Preserve known `0` coefficients for unrelated seeds.

This rule is useful only for index-like casts that produce index values queried
by map-store analysis. It should not try to make arbitrary integer arithmetic
globally affine.

### Gap C: `arith.cmpi` and `arith.andi`

Priority: not required for map-store index contiguity in the inspected cases.

Examples:

- Focused generic and E2E snippets use `arith.cmpi` only to compute the
  yielded mask.
- The BOO sampled convolution cases use `arith.cmpi` and `arith.andi` only to
  compute the yielded mask after `affine.delinearize_index`.

Design note / proposed transfer rule if a future consumer queries mask
dependence:

1. Treat comparison results as non-index boolean facts, not affine index
   values.
2. If both operands are seed-independent, the mask is seed-independent.
3. If either operand depends on a seed, record only boolean dependence or use a
   separate mask-dependency analysis; do not assign affine coefficients.

The current affine seed-dependency analysis can ignore this for yielded index
proofs. Adding coefficient-style interfaces for comparison/logical mask ops
would be misleading unless the query API grows explicit boolean dependence
support.

## Non-Gaps

- `affine.apply`: current support handles the inspected affine offsets,
  constant strides, repeated operands, and mixed-source affine sums. Strict
  contiguity should reject coefficient `2` or multi-seed dependence; that is a
  correct query result, not an op coverage gap.
- `affine.linearize_index`: current static-basis support proves collapse-like
  cases. Dynamic bases should stay conservative unless folded or represented
  symbolically.
- `arith.addi`: current support handles the non-affine-hint rejection test in
  the full generic map-store source. The expression is affine, but strict
  contiguity may still reject it if the same source index influences multiple
  destination dimensions.
- `iree_linalg_ext.yield`: no result interface is needed. The map-store
  consumer should query the yielded index operands directly.

## Linearize/Delinearize Simplification Gaps

The pre-vectorization affine cleanup tests already cover exact inverse pairs,
many-to-one tails, one-to-many tails, unit/trivial bases, and negative cases
for non-disjoint or product-mismatched transforms. Those patterns explain why
simple collapse-like map stores are already analyzable after cleanup.

The remaining real-world snippets are different: they contain bare
`affine.delinearize_index` operations on map-store block arguments, not a
nearby `linearize_index` / `delinearize_index` pair that cleanup can fold.
There is no general canonicalization that can turn
`floordiv` / `mod` results into globally affine expressions. Any useful proof
for these cases needs additional map-store tile range information or a
structured handling of tiled layout splits.

Potential follow-ups:

1. Run the full 550-command BOO source list with `--verify-numerics` and save
   any additional map-store shapes that exercise different index operations.
2. Decide whether data-tiling no-wrap facts belong in the affine
   seed-dependency analysis API, in the map-store contiguity query, or in a
   separate structured tiled-layout recognizer.
3. Add `arith.index_cast` / `arith.index_castui` models before analyzing
   dynamic-basis or workload-derived index expressions.
4. Keep `affine.delinearize_index` conservative unless the implementation can
   prove an exact inverse or no-wrap tile condition.
