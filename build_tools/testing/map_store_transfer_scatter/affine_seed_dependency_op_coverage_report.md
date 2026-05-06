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
are not globally affine in the source linear index. Before treating that as an
analysis failure, re-run this inventory after the pre-vectorization
linearize/delinearize cleanup. For the remaining unsimplified cases, useful
proofs likely need no-wrap/range information from integer range analysis,
alignment information from integer divisibility analysis, or a structured
consumer for tiled-layout splits.

The map-store body operation inventory is only part of the coverage story.
Real BOO cases also implicitly capture SSA values from enclosing control flow,
including `scf.for` induction variables and `scf.forall` induction values.
Those captures should be treated as seed-independent offsets for the
map-store-region seed query unless their own producer state is unknown.

`arith.index_cast` / `arith.index_castui` should be treated as pass-through for
this work: the result gets the same affine seed-dependency state as the
operand. They were not observed inside the representative map-store region
bodies inspected below, including the BOO sample, but they do appear in nearby
data-tiling workload setup and larger ROCm pipeline tests.

Report hygiene note: the target "pre-analysis IR" point is after affine
cleanup and before map-store contiguity analysis. In this work, the concrete
cleanup pass is `iree-codegen-pre-vectorization-affine-index-cleanup`
(`PreVectorizationAffineIndexCleanupPass`). Some inspected artifacts in this
report were captured before that pass was fully positioned in the pipeline, so
the inventory should be refreshed at the target point once pass placement is
finalized.

## Glossary

- BOO: the Turbine convolution benchmark driver workflow documented in
  `build_tools/testing/map_store_transfer_scatter/snippets/boo/README.md`.
- Bead: the workspace issue-tracking item; this report was written for
  `bd-va2`.
- Pre-analysis IR: IR after pre-vectorization affine cleanup and before the
  map-store contiguity analysis or generic vectorization consumes the region
  index expressions. This is the target pipeline state for refreshed
  inventories; see the report hygiene note above for historical captures.

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
buffer loads, linalg producers, checks, or implicit captures from enclosing
regions.

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
rather than yielded index values in the inspected cases. This in-body
inventory must be read together with the implicit-capture slice described in
the next section.

## Implicit Captures and External Producers

The yielded indices are not always computed solely from map-store block
arguments and operations nested textually inside the region. Region bodies can
implicitly capture SSA values from enclosing loops and distributed control
flow. A representative BOO map-store computes one output dimension with:

```mlir
%36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2)>(
  %arg3, %arg9, %arg7)
```

Here `%arg9` is a map-store region block argument, while `%arg3` and `%arg7`
come from enclosing control flow; `%arg7` is an `scf.for` induction variable in
the inspected dump. This example came from `source_index` 3 in
`build_tools/testing/map_store_transfer_scatter/boo_convs_map_store_sample_report.json`;
regenerate the dump with the BOO command in this report to inspect the full
dispatch. Other captures in the BOO sample include constants, dynamic tensor
dimensions, affine values derived before the map store, and `scf.forall`
induction values used to locate distributed slices.

For affine seed-dependency, these captured values should normally be
independent of the map-store region seeds. The implementation must still make
that independence available to transfer functions for mixed expressions such
as `affine.apply`, otherwise an uninitialized captured operand can prevent a
known seed coefficient from propagating. This means the real coverage question
is not only "which ops are inside the map-store body?", but also "which
external producers feed captured operands?" The initial list to verify in real
pipelines is:

- `scf.for` induction variables captured as seed-independent affine offsets.
- `scf.forall` induction values captured as seed-independent distributed tile
  offsets.
- `affine.apply` producers outside the map-store body that compute captured
  tile bases.
- `tensor.dim` / dynamic size values that feed masks or output bounds and
  should be independent for yielded-index proofs.
- Constants and dispatch/workgroup-derived indices that should be independent
  unless a future query intentionally seeds them.

The current BOO sample report records only region-body operation counts. A
follow-up coverage pass should inventory the producer slice for all captured
operands used by yielded indices, not only the operations nested in the
map-store region. A deterministic inventory should walk backward from yielded
index operands, stop at map-store region block arguments and captured values,
and record each captured value's SSA name, defining op, dominance scope, and
expected seed-dependency state (`Known({})`, coefficient map, or unknown).

When the implementation lands, add a minimal lit test to
`compiler/src/iree/compiler/Dialect/Util/Transforms/test/test_affine_seed_dependency_analysis.mlir`
that mixes a map-store block argument with a captured `scf.for` induction
variable in an `affine.apply`. The test should follow the existing
`iree_unregistered.test_affine_seed_dependency` pattern. The `FUTURE-CHECK`
lines below are the expected checks after implicit-capture handling is wired;
before that implementation, this shape is expected to report a conservative
full-unknown lattice result. These snippets are planning artifacts, not current
regression guardrails; when the implementation lands, derive the exact
`CHECK` strings from the `iree-opt --iree-util-test-affine-seed-dependency-analysis`
output instead of hand-editing coefficient strings.

```mlir
// FUTURE-CHECK-LABEL: @map_store_capture_for_iv
func.func @map_store_capture_for_iv(
    %input: tensor<2x2xf32>, %output: tensor<2x4xf32>
) -> tensor<2x4xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %mask = arith.constant true
  %result = scf.for %iv = %c0 to %c2 step %c1
      iter_args(%out = %output) -> (tensor<2x4xf32>) {
    %next = iree_linalg_ext.map_store %input into %out {
    ^bb0(%idx0: index, %idx1: index):
      %mapped = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%idx1, %iv)
      // FUTURE-CHECK: affine_seed_dependency = "seed0 = 0, seed1 = 1"
      "iree_unregistered.test_affine_seed_dependency"(%mapped, %idx0, %idx1)
          : (index, index, index) -> ()
      // FUTURE-CHECK: affine_seed_dependency = "seed0 = 0, seed1 = 0"
      "iree_unregistered.test_affine_seed_dependency"(%iv, %idx0, %idx1)
          : (index, index, index) -> ()
      iree_linalg_ext.yield %idx0, %mapped, %mask : index, index, i1
    } : tensor<2x2xf32> into tensor<2x4xf32> -> tensor<2x4xf32>
    scf.yield %next : tensor<2x4xf32>
  }
  return %result : tensor<2x4xf32>
}
```

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

## Coverage Assessment and Transfer Rules

### Assessment A: `affine.delinearize_index`

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
2. First rely on the pre-vectorization affine cleanup pass to split exact
   `linearize_index` / `delinearize_index` compositions and other simplifiable
   cases. Re-evaluate real dumps after that cleanup before adding special-case
   local transfer rules.
3. If the linear index depends on a seed and no range/alignment/no-wrap fact is
   available, do not assign a numeric coefficient to quotient or remainder
   results. Mark each dependent seed coefficient as `?`, while preserving
   known `0` for unrelated seeds.
4. Use integer range analysis to prove no-wrap conditions such as "the
   seed-dependent part varies only within one static delinearization tile" or
   "the affine expression cannot cross the relevant basis boundary" for the
   vectorized seed range.
5. Use integer divisibility analysis to prove alignment of the seed-independent
   base/offset. For example, if the non-seed component is divisible by a basis
   product and range analysis proves no carry into the next quotient, an
   innermost delinearized result may have coefficient `1` for the seed while
   outer results have coefficient `0`.
6. Dynamic bases should remain per-seed `?` for dependent seeds unless they
   are folded to constants or the analysis is extended to carry symbolic
   coefficients. If all operands and bases are seed-independent, the results
   are independent.

This is partly an interface-precision gap and partly an analysis-context gap:
the current op interface signature does not receive map-store iteration ranges,
vector tile widths, integer ranges, or alignment/divisibility facts. Adding
only a local op model cannot soundly prove the common data-tiling no-wrap
cases.

### Assessment B: casts (`arith.index_cast` and `arith.index_castui`)

Priority: likely needed as missing external-model wiring for broader generated
pipelines, especially when dynamic workload constants or packed layout
dimensions feed map-store index expressions. The intended transfer semantics
for the current map-store contiguity query are simple pass-through.

Examples:

- `compiler/src/iree/compiler/Codegen/LLVMGPU/test/gpu_pipeline_data_tiling.mlir`
  casts a loaded `i32` workload constant to `index` before data-tiling setup.
- `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_vector_distribute_gfx942.mlir`
  contains several nearby `arith.index_cast` / `arith.index_castui` cases.

Design note / proposed transfer rule:

1. Treat these casts as abstract-state pass-through for now: the result gets
   exactly the same affine seed-dependency state as the operand.
2. This is not a general claim that all integer casts preserve mathematical
   value. It is an engineering shortcut for index-like plumbing in the current
   map-store pipelines.
3. "Exactly the same" means copying the whole lattice value, including full
   unknown, per-seed unknown markers, independence, and known coefficient maps;
   the transfer rule does not re-derive integer semantics from the cast.
4. This rule is valid only while the queried map-store relationships do not
   depend on bitwidth-sensitive, wrap-sensitive, or signedness-sensitive
   integer facts.
5. If a later pipeline exposes truncating, wrapping, or sign-changing casts in
   yielded index expressions, revisit this with integer range analysis and add
   negative tests that produce `?` / unknown as appropriate.
   A concrete future negative shape is an index that narrows through
   `i64 -> i32` before round-tripping to `index`, where truncation could change
   the yielded index relationship.

Implementation acceptance should include focused lit cases in
`compiler/src/iree/compiler/Dialect/Util/Transforms/test/test_affine_seed_dependency_analysis.mlir`
showing a known coefficient map forwarded through `arith.index_cast` and
`arith.index_castui`. If truncating or wrapping casts ever enter the yielded
index slice, add a negative case that returns `?` / unknown rather than
silently cloning operand state. This is a closure requirement for the
implementation bead that wires these models, not a requirement for closing this
report-only bead.

Minimal positive test shape, with expected checks after the cast models are
implemented:

```mlir
// FUTURE-CHECK-LABEL: @index_cast_passthrough
util.func @index_cast_passthrough(%idx: index) {
  %i64 = arith.index_cast %idx : index to i64
  %roundtrip = arith.index_cast %i64 : i64 to index
  // FUTURE-CHECK: affine_seed_dependency = "seed0 = 1"
  "iree_unregistered.test_affine_seed_dependency"(%roundtrip, %idx)
      : (index, index) -> ()
  util.return
}

// -----

// FUTURE-CHECK-LABEL: @index_castui_passthrough
util.func @index_castui_passthrough(%idx: index) {
  %i64 = arith.index_castui %idx : index to i64
  %roundtrip = arith.index_castui %i64 : i64 to index
  // FUTURE-CHECK: affine_seed_dependency = "seed0 = 1"
  "iree_unregistered.test_affine_seed_dependency"(%roundtrip, %idx)
      : (index, index) -> ()
  util.return
}
```

### Mask-only ops: `arith.cmpi` and `arith.andi`

Priority: not required for map-store index contiguity in the inspected cases.

Examples:

- Focused generic and E2E snippets use `arith.cmpi` only to compute the
  yielded mask.
- The BOO sampled convolution cases use `arith.cmpi` and `arith.andi` only to
  compute the yielded mask after `affine.delinearize_index`.

No operation interface work is needed for the current yielded-index contiguity
proof. These ops are mask-only in the inspected cases, and the map-store
consumer can query only yielded index operands.

If a future consumer queries mask dependence:

1. Treat comparison results as non-index boolean facts, not affine index
   values.
2. If both operands are seed-independent, the mask is seed-independent.
3. If either operand depends on a seed, record only boolean dependence or use a
   separate mask-dependency analysis; do not assign affine coefficients.

Adding coefficient-style interfaces for comparison/logical mask ops would be
misleading unless the query API grows explicit boolean dependence support.

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
2. Inventory implicit captures for yielded-index operands, including
   `scf.for` induction variables, `scf.forall` induction values, external
   `affine.apply` producers, and dynamic dimension values; use the reduced lit
   test shape described in `Implicit Captures and External Producers`.
3. Re-evaluate `affine.delinearize_index` after the linearize/delinearize
   cleanup pass, then keep remaining cases conservative unless the
   implementation can prove an exact inverse or no-wrap tile condition using
   cleanup, integer range analysis, and/or integer divisibility analysis.
   Decide whether those no-wrap facts belong in the affine seed-dependency
   analysis API, in the map-store contiguity query, or in a separate structured
   tiled-layout recognizer.
4. Implement `arith.index_cast` / `arith.index_castui` as abstract-state
   pass-through, with focused tests for pass-through plumbing and any future
   truncating/wrapping negative case.
