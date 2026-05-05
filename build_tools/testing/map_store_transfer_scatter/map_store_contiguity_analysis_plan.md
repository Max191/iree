# `map_store` contiguity analysis plan

This report classifies the current `map_store` snippet corpus and sketches
analysis options for preserving more contiguous destination dimensions when
lowering tensor-semantics `iree_linalg_ext.map_store` to
`iree_vector_ext.transfer_scatter`. It is the planning artifact for bead
`bd-297` and should be retired or replaced by implementation tests once the
chosen analysis is landed.

The immediate implementation context is the current map-store vectorization
code in
`compiler/src/iree/compiler/Codegen/Interfaces/VectorizableOpInterface.cpp`,
which was added for the prerequisite work on lowering vectorized `map_store`
ops to `transfer_scatter`. That path currently vectorizes the yielded output
indices and mask through a temporary `linalg.generic`, then emits
`transfer_scatter`. It only marks the innermost output dimension contiguous
when that yielded index is exactly the innermost input index plus a constant and
the mask does not depend on the innermost input index. Every other output
dimension is represented by an index vector, even when it is also an
identity/unit-offset input dimension.

`transfer_scatter` can represent each destination dimension independently as:

- a contiguous vector dimension (`AffineDimExpr` in the base map),
- a scattered index vector (`AffineSymbolExpr` in the base map), or
- a broadcast dimension (`0` in the base map).

The `IndexedVectorOpInterface` verifier
(`verifyIndexedVectorOpInterface` in
`compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`)
only allows base-map results that are a dim, symbol, or constant zero.
Therefore, affine expressions such as `d0 * 4 + d1`, `d0 mod 128`, or
`floordiv(d0, 128)` cannot be encoded as contiguous base dimensions directly.
They must either remain scattered, or be proven locally equivalent to a vector
dimension plus a scalar offset for the specific vector tile being emitted.

## Snippet classification

### Focused generic-vectorization snippets

| Function | Pattern | Preservable input dims | Notes |
| --- | --- | --- | --- |
| `map_store` | rank-3 identity copy, `(i, j, k) -> (i, j, k)` | `i`, `j`, `k` | All yielded output dims are exact block arguments. Current implementation only preserves `k`. |
| `no_vectorize_map_store_dynamic` | rank-1 identity with dynamic input shape | none until vectorization legality changes | Static-shape guard prevents vectorization before contiguity matters. |
| `map_store_f4_multiple_of_byte` | rank-2 identity sub-byte copy, `(i, j) -> (i, j)` | `i`, `j` after byte-pack legality | The inner extent is byte-aligned, so identity dims are valid candidates. |
| `map_store_f4_not_multiple_of_byte` | rank-2 identity sub-byte copy with one inner f4 element | none for current vectorization | Fails byte-multiple legality. Contiguity proof alone is insufficient. |
| `map_store_f4_unit_stride` | unit-offset slice/write, `(i, j) -> (i, j + 2)` | `i`, `j` | `j` can be represented by destination offset `2`; `i` is also an identity candidate. |
| `map_store_f4_not_unit_stride` | strided write, `(i, j) -> (i, 2*j)` | `i` only | `2*j` is not a contiguous dim in `transfer_scatter`. |
| `map_store_f4_not_index_applied_multiple_times` | repeated-index affine write, `(i, j) -> (i, j + j)` | `i` only | Repeated use of `j` changes the stride to `2`. |
| `map_store_f4_mask_depends_on_inner_index` | identity write with lane-dependent mask | `i`, `j` for metadata; needs sub-byte mask policy | The map is contiguous, but current f4 legality rejects inner-index-dependent masks. |

### Focused e2e snippets

| Function | Pattern | Preservable input dims | Notes |
| --- | --- | --- | --- |
| `copy_like` | rank-3 identity copy | `i`, `j`, `k` | Same shape and same input block arguments. |
| `collapse_shape_like` | rank-2 to rank-1 `affine.linearize_index`, `(i, j) -> i * 4 + j` | none directly in rank-1 base map | The output dim is a composite affine expression. This is a contiguous flattened store semantically, but not expressible as a single dim expr without shape-cast/linearization support. |
| `expand_shape_shape_like` | rank-1 to rank-2 `affine.delinearize_index`, `i -> (i / 4, i mod 4)` | none globally | `i mod 4` is locally contiguous only within no-wrap tiles; `i / 4` is outer tile selection. |
| `extract_slice_like` | rank-1 identity with mask, `i -> i` | `i` | Mask limits active lanes; contiguity metadata remains straightforward for full-byte elements. |

### GPU snippets

| File/function | Pattern | Preservable input dims | Notes |
| --- | --- | --- | --- |
| `gpu/rocdl_tile_and_fuse_data_tiling_map_store_pre.mlir` / `map_store` | 2D data tiling, `(m, n) -> (m / 128, n / 128, m mod 128, n mod 128)` | inner tile leaves `m mod 128`, `n mod 128` are locally preservable; no dims are globally exact | Requires proving the vector tile does not cross a 128-element boundary and deriving scalar offsets from tile origins. |
| `gpu/rocdl_matmul_map_store_pre_vector_distribute.mlir` / `matmul_map_store` | packed matmul layout via nested delinearization | small packed leaves such as `m mod 4` and `n mod 4` are locally preservable; no dims are globally exact | The packed layout permutes and splits both input dims. Proving useful contiguous groups requires decomposition of nested delinearize trees plus tile-bound/no-wrap facts. |

The BOO convolution-snippet workflow is documented in
`build_tools/testing/map_store_transfer_scatter/snippets/boo/README.md`. Rerun
this classification when committed `case_<N>_*.mlir` snippets are added.

## Candidate proof strategies

### Option A: direct block-argument/unit-offset matcher

Generalize the existing innermost-only helper to every output dimension:

1. For each yielded output index, prove it is `inputDim + constant` for exactly
   one map-store input block argument.
2. Reject repeated use as a contiguous base-map dim only when it would violate
   `verifyIndexedVectorOpInterface` constraints or duplicate vector-dim
   positions in the base map. Do not confuse this with the legal case where the
   same input SSA value is scattered for one output dimension and also used as
   the single contiguous dim for another output dimension.
3. Emit that output dimension as an `AffineDimExpr(inputDim)` and store the
   constant in the corresponding scalar offset.
4. Emit all remaining output dimensions as index vectors, as today.

This option covers the identity, unit-offset, masked full-byte, and outer-dim
identity parts of the focused corpus. It does not cover `linearize_index`,
`delinearize_index`, modulo, or tile-local contiguity.

### Option B: affine expression classifier before vectorization

Classify yielded output indices structurally before lowering
`affine.linearize_index` and `affine.delinearize_index` in the temporary
`linalg.generic`:

- `blockArg + constant`: globally contiguous.
- `blockArg * stride + constant`, repeated block args, or non-constant offsets:
  scattered.
- `linearize_index` of a suffix of input dims: potentially contiguous after
  output shape-cast support, but not directly encodable in current
  `transfer_scatter`.
- `delinearize_index` result leaves: locally contiguous if the vector tile is
  known not to wrap the corresponding basis.

This preserves high-level intent that is erased when delinearize/linearize ops
are lowered to arithmetic. It also gives clearer diagnostics and test names.

### Option C: tile-local no-wrap analysis

For GPU packed/tiled snippets, prove that a delinearized leaf is equivalent to
`inputDim + scalarOffset` over the vector tile:

1. Recover a decomposition tree for each yielded index, including nested
   `affine.delinearize_index` bases.
2. For each leaf, compute its period and stride within the original input dim.
3. Use the vector tile's lower bound, extent, and input-dim step to prove the
   tile stays within one period. For example, `m mod 128` is equivalent to
   `m + offset` if all active lanes are within one 128-wide tile.
4. Materialize the leaf as a contiguous dim in the base map and put the tile
   origin or modulo base in scalar offsets.
5. Keep outer quotient dims and any wrapping leaf as scattered index vectors.

This is the first option that can make the committed GPU snippets materially
better. It needs facts from vector tile selection or loop/tile bounds; it should
not rely on incidental post-vectorization arithmetic shapes.

### Option D: use dataflow facts from vector tile/materialization analysis

PR <https://github.com/iree-org/iree/pull/24318> is a useful planning reference
because it wires integer divisibility information into codegen tile-size
analysis for im2col vectorization. The reusable idea is to make divisibility,
alignment, and tile-bound facts available to vectorization consumers instead of
rediscovering those facts from local affine syntax. This report should not
depend on that PR or its exact implementation, but the same kind of facts would
be useful here:

- divisibility/alignment of loop lower bounds,
- vector tile extents,
- whether an offset expression has converged before vector-size decisions use
  it, and
- whether an analysis can answer no-wrap questions mechanically instead of via
  ad hoc affine-op matching.

For `map_store`, those facts can prove that a modulo/delinearize leaf remains
within one contiguous output tile. If the dataflow infrastructure does not
expose analysis dependencies yet, keep this as an optional enhancement rather
than a blocker for Option A.

## Proposed staging

1. Implement Option A first. It is local to map-store vectorization, easy to
   test, and covers all exact/unit-offset dimensions in the focused corpus.
2. Add tests that assert base-map dim expressions for all identity/unit-offset
   dimensions, not only the innermost one.
3. Add negative tests for stride, repeated-index, and non-constant-offset
   expressions to keep them scattered.
4. Keep the existing multi-use behavior covered by
   `map_store_f32_inner_dim_used_by_multiple_output_dims` in
   `compiler/src/iree/compiler/Codegen/Common/test/generic_vectorization_map_store.mlir`:
   one output can scatter an input SSA value while another output uses that same
   input dim as the contiguous base-map dim.
5. Prototype Option B classification while retaining the current fallback path
   that vectorizes all non-contiguous indices.
6. Keep tile-local no-wrap facts produced by tiling/configuration or
   materialized tile-size analysis; map-store vectorization should consume only
   a narrow contract: vector extent, tile lower bound, and divisibility/no-wrap
   answers for each candidate input dimension.
7. Add GPU-oriented tests for Option C only after the implementation has a
   stable source of tile lower-bound/extents/divisibility facts.

## Undefined or needs discussion

- **Mask policy:** Full-byte element types can keep contiguous metadata with a
  lane-dependent mask. Sub-byte types may still need the current stricter rule
  to avoid partially updating packed bytes. The exact legality should be stated
  in map-store vectorization tests.
- **Uniqueness/data races:** `transfer_scatter` requires unique destination
  indices. Contiguity analysis should not imply uniqueness for scattered dims;
  the existing map-store semantics or a separate proof must cover it.
- **Duplicate contiguous dim use:** Existing tests allow the innermost input dim
  to be both scattered in one output dim and contiguous in another
  (`map_store_f32_inner_dim_used_by_multiple_output_dims`). The broader
  analysis should specify whether duplicate contiguous use in two base-map
  dimensions is ever legal; current canonicalization treats duplicate dim exprs
  as not fully contiguous.
- **Flattened linearized writes:** `collapse_shape_like` is semantically a
  contiguous flattened store, but current `transfer_scatter` base maps cannot
  encode `i * extent + j`. Handling this likely needs shape-cast-aware
  vectorization or a separate linearized transfer representation.
- **Tile-local facts:** GPU delinearize cases need tile-origin, vector extent,
  and no-wrap/divisibility facts. The owner of those facts should be clarified:
  vectorization can consume them, but tiling/configuration likely owns producing
  them.
- **Dynamic shapes:** Current map-store vectorization rejects dynamic input
  shapes. A future dynamic path needs runtime no-wrap/byte-pack conditions or
  conservative fallback.
- **BOO corpus evolution:** No BOO snippet cases are committed here yet. Once
  added, classify them with the same taxonomy and record whether they are exact,
  affine-strided, linearized, delinearized, or mixed.

## Lightweight validation checklist for this report

- Confirm every committed snippet file contains `iree_linalg_ext.map_store`.
- Grep the report for every committed snippet function name.
- Grep `preserveInnermostContiguousDim` and `getConstantUnitOffset` in
  `VectorizableOpInterface.cpp` to find the current `MapStoreOp` vectorization
  region that preserves only the innermost unit-offset dim.
- Confirm referenced local paths exist.
- Confirm the PR link is present only as a planning resource, not as a required
  dependency.
