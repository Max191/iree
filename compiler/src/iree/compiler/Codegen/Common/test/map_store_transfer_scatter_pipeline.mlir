// RUN: iree-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-pre-vectorization-affine-index-cleanup,iree-codegen-annotate-map-store-contiguous-dim-hints,iree-codegen-generic-vectorization{enable-vector-masking=false vectorize-map-store=true}))" \
// RUN:   --split-input-file %s | FileCheck %s

// These tests keep vector masking disabled to isolate the map_store indexing
// prelude from tile-size materialization effects.

// The pre-vectorization pipeline derives an affine indexing map and keeps the
// dynamic offset that is independent of map_store index block arguments as a
// transfer_scatter base offset.
func.func @map_store_pipeline_affine_offset(
    %input: tensor<2x3xf32>, %output: tensor<?x?xf32>, %offset: index
) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %c4 = arith.constant 4 : index
      %linear = arith.muli %idx0, %c4 : index
      %base = arith.addi %linear, %idx1 : index
      %with_offset = arith.addi %base, %offset : index
      iree_linalg_ext.yield %idx0, %with_offset, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-DAG:   #[[$AFFINE_OFFSET_MAP:.+]] = affine_map<(d0, d1) -> (d0, d0 * 4 + d1)>
// CHECK-LABEL: @map_store_pipeline_affine_offset
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OFFSET:[a-zA-Z0-9_]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-NOT:   iree_linalg_ext.map_store
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//   CHECK-NOT:   iree_linalg_ext.map_store
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]]
//  CHECK-SAME:     [%[[C0]], %[[OFFSET]]]
//  CHECK-SAME:     indexing_maps = [#[[$AFFINE_OFFSET_MAP]]]
//   CHECK-NOT:   iree_linalg_ext.map_store
//       CHECK:     : vector<2x3xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
//       CHECK:   return %[[SCATTER]] : tensor<?x?xf32>

// -----

// Non-affine map_store index relationships are kept as transfer_scatter symbols
// while known affine relationships remain in the indexing map.
func.func @map_store_pipeline_unknown_symbol(
    %input: tensor<2x3xf32>, %output: tensor<6xf32>
) -> tensor<6xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %unknown = arith.muli %idx0, %idx1 : index
      iree_linalg_ext.yield %unknown, %mask : index, i1
  } : tensor<2x3xf32> into tensor<6xf32> -> tensor<6xf32>
  return %0 : tensor<6xf32>
}
// CHECK-DAG:   #[[$UNKNOWN_SYMBOL_MAP:.+]] = affine_map<(d0, d1)[s0] -> (s0)>
// CHECK-DAG:   #[[$UNKNOWN_SYMBOL_ID_MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0, d1)>
// CHECK-LABEL: @map_store_pipeline_unknown_symbol
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[INDEX_VEC:.+]] = arith.muli {{.*}} : vector<2x3xindex>
//   CHECK-NOT:   iree_linalg_ext.map_store
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//   CHECK-NOT:   iree_linalg_ext.map_store
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]]
//  CHECK-SAME:     [%[[INDEX_VEC]] : vector<2x3xindex>]
//  CHECK-SAME:     indexing_maps = [#[[$UNKNOWN_SYMBOL_MAP]], #[[$UNKNOWN_SYMBOL_ID_MAP]]]
//   CHECK-NOT:   iree_linalg_ext.map_store
//       CHECK:     : vector<2x3xf32>, tensor<6xf32> -> tensor<6xf32>
//       CHECK:   return %[[SCATTER]] : tensor<6xf32>
