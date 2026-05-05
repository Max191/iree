// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=false vectorize-map-store=true}))" --split-input-file %s | FileCheck %s

func.func @map_store(
    %input: tensor<4x16x64xf32>, %output: tensor<4x16x64xf32>
) -> tensor<4x16x64xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index, %idx2: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %idx2, %mask : index, index, index, i1
  } : tensor<4x16x64xf32> into tensor<4x16x64xf32> -> tensor<4x16x64xf32>
  return %0 : tensor<4x16x64xf32>
}
// CHECK-LABEL: @map_store
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]
//  CHECK-SAME:     : vector<4x16x64xf32>, tensor<4x16x64xf32> -> tensor<4x16x64xf32>
//       CHECK:   return %[[SCATTER]] : tensor<4x16x64xf32>

// -----

// Tiny non-contiguous tiles stay on the memref fallback path to avoid
// workgroup-staged tensor transfer_scatter overhead.
func.func @map_store_f32_not_unit_stride(
    %input: tensor<2x2xf32>, %output: tensor<2x4xf32>
) -> tensor<2x4xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x2xf32> into tensor<2x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
// CHECK-LABEL: @map_store_f32_not_unit_stride
//       CHECK:   iree_linalg_ext.map_store
//   CHECK-NOT:   transfer_scatter

// -----

func.func @map_store_f32_noncontiguous_eight_element_tile(
    %input: tensor<8xf32>, %output: tensor<16xf32>
) -> tensor<16xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%idx0)
      iree_linalg_ext.yield %1, %mask : index, i1
  } : tensor<8xf32> into tensor<16xf32> -> tensor<16xf32>
  return %0 : tensor<16xf32>
}
// CHECK-LABEL: @map_store_f32_noncontiguous_eight_element_tile
//       CHECK:   iree_linalg_ext.map_store
//   CHECK-NOT:   transfer_scatter

// -----

func.func @map_store_f32_noncontiguous_nine_element_tile(
    %input: tensor<9xf32>, %output: tensor<18xf32>
) -> tensor<18xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%idx0)
      iree_linalg_ext.yield %1, %mask : index, i1
  } : tensor<9xf32> into tensor<18xf32> -> tensor<18xf32>
  return %0 : tensor<18xf32>
}
// CHECK-LABEL: @map_store_f32_noncontiguous_nine_element_tile
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]]]
//  CHECK-SAME:     vector<9xindex>
//  CHECK-SAME:     : vector<9xf32>, tensor<18xf32> -> tensor<18xf32>
//       CHECK:   return %[[SCATTER]] : tensor<18xf32>

// -----

func.func @map_store_f32_tiny_noncontiguous_explicit_hint(
    %input: tensor<2x4xf32>, %output: tensor<2x8xf32>
) -> tensor<2x8xf32> {
  %0 = iree_linalg_ext.map_store {contiguous_dim_hints = array<i64: 0, 0>}
      %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x4xf32> into tensor<2x8xf32> -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
// CHECK-LABEL: @map_store_f32_tiny_noncontiguous_explicit_hint
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C0]]]
//  CHECK-SAME:     : vector<2x4xf32>, tensor<2x8xf32> -> tensor<2x8xf32>
//       CHECK:   return %[[SCATTER]] : tensor<2x8xf32>

// -----

func.func @map_store_f32_not_unit_stride_large_tile(
    %input: tensor<2x2x4xf32>, %output: tensor<2x2x8xf32>
) -> tensor<2x2x8xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index, %idx2: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%idx2)
      iree_linalg_ext.yield %idx0, %idx1, %1, %mask
          : index, index, index, i1
  } : tensor<2x2x4xf32> into tensor<2x2x8xf32> -> tensor<2x2x8xf32>
  return %0 : tensor<2x2x8xf32>
}
// CHECK-LABEL: @map_store_f32_not_unit_stride_large_tile
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]
//  CHECK-SAME:     vector<4xindex>
//  CHECK-SAME:     : vector<2x2x4xf32>, tensor<2x2x8xf32> -> tensor<2x2x8xf32>
//       CHECK:   return %[[SCATTER]] : tensor<2x2x8xf32>

// -----

// The innermost input dim is still represented as a contiguous transfer dim
// when another output dim also uses it as a scattered index.
func.func @map_store_f32_inner_dim_used_by_multiple_output_dims(
    %input: tensor<2x3xf32>, %output: tensor<2x3x3xf32>
) -> tensor<2x3x3xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %idx1, %mask : index, index, index, i1
  } : tensor<2x3xf32> into tensor<2x3x3xf32> -> tensor<2x3x3xf32>
  return %0 : tensor<2x3x3xf32>
}
// CHECK-LABEL: @map_store_f32_inner_dim_used_by_multiple_output_dims
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]
//  CHECK-SAME:     vector<3xindex>
//  CHECK-SAME:     : vector<2x3xf32>, tensor<2x3x3xf32> -> tensor<2x3x3xf32>
//       CHECK:   return %[[SCATTER]] : tensor<2x3x3xf32>

// -----

// A non-constant offset on the innermost output dim must stay scattered.
func.func @map_store_f32_inner_offset_depends_on_outer_dim(
    %input: tensor<2x3xf32>, %output: tensor<2x5xf32>
) -> tensor<2x5xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%idx0, %idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x5xf32> -> tensor<2x5xf32>
  return %0 : tensor<2x5xf32>
}
// CHECK-LABEL: @map_store_f32_inner_offset_depends_on_outer_dim
//       CHECK:   iree_linalg_ext.map_store
//   CHECK-NOT:   transfer_scatter

// -----

// Explicit hints augment, rather than replace, the existing innermost
// contiguous-dim preservation. Here the hint marks the outer dim contiguous,
// and the legacy innermost check still preserves the inner dim.
func.func @map_store_f32_explicit_hint_keeps_legacy_innermost(
    %input: tensor<2x3xf32>, %output: tensor<2x3xf32>
) -> tensor<2x3xf32> {
  %0 = iree_linalg_ext.map_store {contiguous_dim_hints = array<i64: 0, 0>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
// CHECK-DAG: #[[$MIXED_HINT_BASE_MAP:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 + s0, d1 + s1)>
// CHECK-DAG: #[[$MIXED_HINT_INDEX_MAP_0:.+]] = affine_map<(d0, d1)[s0, s1] -> (d1)>
// CHECK-DAG: #[[$MIXED_HINT_INDEX_MAP_1:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0)>
// CHECK-LABEL: @map_store_f32_explicit_hint_keeps_legacy_innermost
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[ZERO_INNER:.+]] = arith.constant dense<0> : vector<3xindex>
//   CHECK-DAG:   %[[ZERO_OUTER:.+]] = arith.constant dense<0> : vector<2xindex>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C0]]] [%[[ZERO_INNER]], %[[ZERO_OUTER]] : vector<3xindex>, vector<2xindex>]
//  CHECK-SAME:     indexing_maps = [#[[$MIXED_HINT_BASE_MAP]], #[[$MIXED_HINT_INDEX_MAP_0]], #[[$MIXED_HINT_INDEX_MAP_1]]]
//  CHECK-SAME:     : vector<2x3xf32>, tensor<2x3xf32> -> tensor<2x3xf32>
//       CHECK:   return %[[SCATTER]] : tensor<2x3xf32>

// -----

// Explicit hints preserve the legacy innermost constant offset path. The hint
// marks the outer dim contiguous, and the inner affine offset must still be
// threaded through the transfer_scatter base expression.
func.func @map_store_f32_explicit_hint_keeps_inner_constant_offset(
    %input: tensor<2x3xf32>, %output: tensor<2x4xf32>
) -> tensor<2x4xf32> {
  %0 = iree_linalg_ext.map_store {contiguous_dim_hints = array<i64: 0, 0>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
// CHECK-DAG: #[[$MIXED_HINT_OFFSET_BASE_MAP:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 + s0, d1 + s1)>
// CHECK-DAG: #[[$MIXED_HINT_OFFSET_INDEX_MAP_0:.+]] = affine_map<(d0, d1)[s0, s1] -> (d1)>
// CHECK-DAG: #[[$MIXED_HINT_OFFSET_INDEX_MAP_1:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0)>
// CHECK-LABEL: @map_store_f32_explicit_hint_keeps_inner_constant_offset
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[ZERO_OUTER:.+]] = arith.constant dense<0> : vector<3xindex>
//   CHECK-DAG:   %[[ONE_INNER:.+]] = arith.constant dense<1> : vector<2xindex>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C0]]] [%[[ZERO_OUTER]], %[[ONE_INNER]] : vector<3xindex>, vector<2xindex>]
//  CHECK-SAME:     indexing_maps = [#[[$MIXED_HINT_OFFSET_BASE_MAP]], #[[$MIXED_HINT_OFFSET_INDEX_MAP_0]], #[[$MIXED_HINT_OFFSET_INDEX_MAP_1]]]
//  CHECK-SAME:     : vector<2x3xf32>, tensor<2x4xf32> -> tensor<2x4xf32>
//       CHECK:   return %[[SCATTER]] : tensor<2x4xf32>

// -----

// A contiguity hint lets the vectorizer preserve the inner source dim as a
// contiguous transfer_scatter dim even when the output dim also has an outer
// scattered base component.
func.func @map_store_f32_hinted_inner_contiguous_with_outer_base(
    %input: tensor<2x3xf32>, %output: tensor<2x5xf32>
) -> tensor<2x5xf32> {
  %0 = iree_linalg_ext.map_store {contiguous_dim_hints = array<i64: 1, 1>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%idx0, %idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x5xf32> -> tensor<2x5xf32>
  return %0 : tensor<2x5xf32>
}
// CHECK-DAG: #[[$HINTED_BASE_MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0, d1 + s0)>
// CHECK-DAG: #[[$HINTED_INDEX_MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0)>
// CHECK-LABEL: @map_store_f32_hinted_inner_contiguous_with_outer_base
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[INDEX:.+]] = vector.shape_cast
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C0]]] [%[[INDEX]] : vector<2xindex>]
//  CHECK-SAME:     indexing_maps = [#[[$HINTED_BASE_MAP]], #[[$HINTED_INDEX_MAP]]]
//  CHECK-SAME:     : vector<2x3xf32>, tensor<2x5xf32> -> tensor<2x5xf32>
//       CHECK:   return %[[SCATTER]] : tensor<2x5xf32>

// -----

// Hints still preserve and plumb a non-trivial mask through transfer_scatter.
func.func @map_store_f32_hinted_inner_contiguous_with_mask(
    %input: tensor<2x3xf32>, %output: tensor<2x5xf32>
) -> tensor<2x5xf32> {
  %0 = iree_linalg_ext.map_store {contiguous_dim_hints = array<i64: 1, 1>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %c1 = arith.constant 1 : index
      %mask = arith.cmpi uge, %idx1, %c1 : index
      %1 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%idx0, %idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x5xf32> -> tensor<2x5xf32>
  return %0 : tensor<2x5xf32>
}
// CHECK-DAG: #[[$HINTED_MASK_BASE_MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0, d1 + s0)>
// CHECK-DAG: #[[$HINTED_MASK_INDEX_MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0)>
// CHECK-DAG: #[[$HINTED_MASK_MASK_MAP:.+]] = affine_map<(d0, d1)[s0] -> (d1)>
// CHECK-LABEL: @map_store_f32_hinted_inner_contiguous_with_mask
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[MASK:.+]] = arith.cmpi uge
//       CHECK:   %[[INDEX:.+]] = vector.shape_cast
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C0]]] [%[[INDEX]] : vector<2xindex>], %[[MASK]]
//  CHECK-SAME:     indexing_maps = [#[[$HINTED_MASK_BASE_MAP]], #[[$HINTED_MASK_INDEX_MAP]], #[[$HINTED_MASK_MASK_MAP]]]
//  CHECK-SAME:     : vector<2x3xf32>, tensor<2x5xf32>, vector<3xi1> -> tensor<2x5xf32>
//       CHECK:   return %[[SCATTER]] : tensor<2x5xf32>

// -----

// Multiple hinted input dims can contribute to the same output dim. This keeps
// both dims in the transfer_scatter base map and removes the gathered index vec.
func.func @map_store_f32_multiple_hints_same_output_dim(
    %input: tensor<2x3xf32>, %output: tensor<4xf32>
) -> tensor<4xf32> {
  %0 = iree_linalg_ext.map_store {contiguous_dim_hints = array<i64: 0, 0, 1, 0>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%idx0, %idx1)
      iree_linalg_ext.yield %1, %mask : index, i1
  } : tensor<2x3xf32> into tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-DAG: #[[$STACKED_HINT_BASE_MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: @map_store_f32_multiple_hints_same_output_dim
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]]]
//  CHECK-SAME:     indexing_maps = [#[[$STACKED_HINT_BASE_MAP]]]
//  CHECK-SAME:     : vector<2x3xf32>, tensor<4xf32> -> tensor<4xf32>
//       CHECK:   return %[[SCATTER]] : tensor<4xf32>

// -----

// This hint is structurally valid but semantically false: the hinted input dim
// has coefficient 2 in the hinted output dim, so validation rejects it before
// transfer_scatter construction. Expected match failure: "hinted contiguous
// input dim must have unit coefficient".
func.func @map_store_f32_reject_bad_contiguous_hint(
    %input: tensor<2x3xf32>, %output: tensor<2x8xf32>
) -> tensor<2x8xf32> {
  %0 = iree_linalg_ext.map_store {contiguous_dim_hints = array<i64: 1, 1>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%idx0, %idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x8xf32> -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
// CHECK-LABEL: @map_store_f32_reject_bad_contiguous_hint
//       CHECK:   iree_linalg_ext.map_store
//   CHECK-NOT:   transfer_scatter

// -----

// This hint is structurally valid but semantically false: the hinted input dim
// contributes to two output dims, so it is not contiguous in only one output
// dimension. Expected match failure: "hinted contiguous input dim must affect
// only one output dim".
func.func @map_store_f32_reject_hint_affects_multiple_outputs(
    %input: tensor<2x3xf32>, %output: tensor<3x3xf32>
) -> tensor<3x3xf32> {
  %0 = iree_linalg_ext.map_store {contiguous_dim_hints = array<i64: 1, 1>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx1, %idx1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<3x3xf32> -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
// CHECK-LABEL: @map_store_f32_reject_hint_affects_multiple_outputs
//       CHECK:   iree_linalg_ext.map_store
//   CHECK-NOT:   transfer_scatter

// -----

// This hint is structurally valid but validation can only prove affine output
// index expressions. Non-affine index arithmetic keeps the map_store intact.
// Expected match failure: "contiguous dim hints require affine output index
// expressions".
func.func @map_store_f32_reject_non_affine_hint(
    %input: tensor<2x3xf32>, %output: tensor<2x5xf32>
) -> tensor<2x5xf32> {
  %0 = iree_linalg_ext.map_store {contiguous_dim_hints = array<i64: 1, 1>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = arith.addi %idx0, %idx1 : index
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x5xf32> -> tensor<2x5xf32>
  return %0 : tensor<2x5xf32>
}
// CHECK-LABEL: @map_store_f32_reject_non_affine_hint
//       CHECK:   iree_linalg_ext.map_store
//   CHECK-NOT:   transfer_scatter

// -----

// Full-byte map_store with a mask depending on the inner dim stays on the
// post-bufferization decomposition path.
func.func @map_store_f32_mask_depends_on_inner_index(
    %input: tensor<2x2xf32>, %output: tensor<2x2xf32>
) -> tensor<2x2xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %c1 = arith.constant 1 : index
      %mask = arith.cmpi uge, %idx1, %c1 : index
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x2xf32> into tensor<2x2xf32> -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: @map_store_f32_mask_depends_on_inner_index
//       CHECK:   iree_linalg_ext.map_store
//   CHECK-NOT:   transfer_scatter

// -----

func.func @no_vectorize_map_store_dynamic(
    %input: tensor<?xf32>, %output: tensor<64xf32>
) -> tensor<64xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %mask : index, i1
  } : tensor<?xf32> into tensor<64xf32> -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
// CHECK-LABEL: @no_vectorize_map_store_dynamic
//   CHECK-NOT:   vector

// -----

func.func @map_store_f4_multiple_of_byte(
    %input: tensor<2x2xf4E2M1FN>, %output: tensor<2x2xf4E2M1FN>
) -> tensor<2x2xf4E2M1FN> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x2xf4E2M1FN> into tensor<2x2xf4E2M1FN> -> tensor<2x2xf4E2M1FN>
  return %0 : tensor<2x2xf4E2M1FN>
}
// CHECK-LABEL: @map_store_f4_multiple_of_byte
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C0]]]
//  CHECK-SAME:     : vector<2x2xf4E2M1FN>, tensor<2x2xf4E2M1FN> -> tensor<2x2xf4E2M1FN>
//       CHECK:   return %[[SCATTER]] : tensor<2x2xf4E2M1FN>

// -----

func.func @map_store_f4_not_multiple_of_byte(
    %input: tensor<2x1xf4E2M1FN>, %output: tensor<2x2xf4E2M1FN>
) -> tensor<2x2xf4E2M1FN> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x1xf4E2M1FN> into tensor<2x2xf4E2M1FN> -> tensor<2x2xf4E2M1FN>
  return %0 : tensor<2x2xf4E2M1FN>
}
// CHECK-LABEL: @map_store_f4_not_multiple_of_byte
//   CHECK-NOT:   vector

// -----

func.func @map_store_f4_unit_stride(
    %input: tensor<2x2xf4E2M1FN>, %output: tensor<2x4xf4E2M1FN>
) -> tensor<2x4xf4E2M1FN> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 + 2)>(%idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x2xf4E2M1FN> into tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
  return %0 : tensor<2x4xf4E2M1FN>
}
// CHECK-LABEL: @map_store_f4_unit_stride
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C2]]]
//  CHECK-SAME:     : vector<2x2xf4E2M1FN>, tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
//       CHECK:   return %[[SCATTER]] : tensor<2x4xf4E2M1FN>

// -----

func.func @map_store_f4_not_unit_stride(
    %input: tensor<2x2xf4E2M1FN>, %output: tensor<2x4xf4E2M1FN>
) -> tensor<2x4xf4E2M1FN> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x2xf4E2M1FN> into tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
  return %0 : tensor<2x4xf4E2M1FN>
}
// CHECK-LABEL: @map_store_f4_not_unit_stride
//   CHECK-NOT:   vector

// -----

func.func @map_store_f4_not_index_applied_multiple_times(
    %input: tensor<2x2xf4E2M1FN>, %output: tensor<2x4xf4E2M1FN>
) -> tensor<2x4xf4E2M1FN> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%idx1, %idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x2xf4E2M1FN> into tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
  return %0 : tensor<2x4xf4E2M1FN>
}
// CHECK-LABEL: @map_store_f4_not_index_applied_multiple_times
//   CHECK-NOT:   vector

// -----

func.func @map_store_f4_mask_depends_on_inner_index(
    %input: tensor<2x2xf4E2M1FN>, %output: tensor<2x4xf4E2M1FN>
) -> tensor<2x4xf4E2M1FN> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %c1 = arith.constant 1 : index
      %mask = arith.cmpi uge, %idx1, %c1 : index
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x2xf4E2M1FN> into tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
  return %0 : tensor<2x4xf4E2M1FN>
}
// CHECK-LABEL: @map_store_f4_mask_depends_on_inner_index
//   CHECK-NOT:   vector
