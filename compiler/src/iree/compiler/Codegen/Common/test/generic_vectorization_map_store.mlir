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
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]]
//       CHECK:     : vector<4x16x64xf32>, tensor<4x16x64xf32> -> tensor<4x16x64xf32>
//       CHECK:   return %[[SCATTER]] : tensor<4x16x64xf32>

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
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]]
//       CHECK:     : vector<2x2xf4E2M1FN>, tensor<2x2xf4E2M1FN> -> tensor<2x2xf4E2M1FN>
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
//       CHECK:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][{{.*}}, %[[C2]]]
//       CHECK:     : vector<2x2xf4E2M1FN>, tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
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

// -----

// A hinted affine map carries only the vector-dimension contribution. The
// seed-independent `%offset + 7` residual is materialized as the scatter offset.
func.func @map_store_hinted_affine_offset(
    %input: tensor<2x3xf32>, %output: tensor<?x?xf32>, %offset: index
) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.map_store {transfer_scatter_indexing_map = affine_map<(d0, d1) -> (d0, d0 * 4 + d1)>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %c4 = arith.constant 4 : index
      %linear = arith.muli %idx0, %c4 : index
      %base = arith.addi %linear, %idx1 : index
      %with_offset = arith.addi %base, %offset : index
      %c7 = arith.constant 7 : index
      %out1 = arith.addi %with_offset, %c7 : index
      iree_linalg_ext.yield %idx0, %out1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-DAG:   #[[$AFFINE_OFFSET_MAP:.+]] = affine_map<(d0, d1) -> (d0, d0 * 4 + d1)>
// CHECK-LABEL: @map_store_hinted_affine_offset
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//   CHECK-DAG:   %[[OFFSET:.+]] = arith.addi {{.*}}, %[[C7]] : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][{{.*}}, %[[OFFSET]]]
//       CHECK-SAME: indexing_maps = [#[[$AFFINE_OFFSET_MAP]]{{.*}}]
//       CHECK:     : vector<2x3xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
//       CHECK:   return %[[SCATTER]] : tensor<?x?xf32>

// -----

// A hinted symbol result asks vectorization to compute that full output index
// as an index vector while preserving known affine dimensions in the base map.
func.func @map_store_hinted_unknown_symbol(
    %input: tensor<2x3xf32>, %output: tensor<2x6xf32>
) -> tensor<2x6xf32> {
  %0 = iree_linalg_ext.map_store {transfer_scatter_indexing_map = affine_map<(d0, d1)[s0] -> (d0, s0)>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %sum = arith.addi %idx0, %idx1 : index
      iree_linalg_ext.yield %idx0, %sum, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x6xf32> -> tensor<2x6xf32>
  return %0 : tensor<2x6xf32>
}
// CHECK-DAG:   #[[$UNKNOWN_SYMBOL_MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0, s0)>
// CHECK-LABEL: @map_store_hinted_unknown_symbol
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[INDEX_VEC:.+]] = arith.addi
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]]
//  CHECK-SAME:     [%[[INDEX_VEC]] : vector<2x3xindex>]
//       CHECK-SAME: indexing_maps = [#[[$UNKNOWN_SYMBOL_MAP]], {{.*}}]
//       CHECK:     : vector<2x3xf32>, tensor<2x6xf32> -> tensor<2x6xf32>
//       CHECK:   return %[[SCATTER]] : tensor<2x6xf32>

// -----

func.func @map_store_hinted_wrong_rank(
    %input: tensor<2x3xf32>, %output: tensor<2x3xf32>
) -> tensor<2x3xf32> {
  %0 = iree_linalg_ext.map_store {transfer_scatter_indexing_map = affine_map<(d0) -> (d0, d0)>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
// CHECK-LABEL: @map_store_hinted_wrong_rank
//   CHECK-NOT: iree_vector_ext.transfer_scatter
//       CHECK: iree_linalg_ext.map_store

// -----

func.func @map_store_hinted_non_result_symbol_use(
    %input: tensor<2x3xf32>, %output: tensor<2x6xf32>
) -> tensor<2x6xf32> {
  %0 = iree_linalg_ext.map_store {transfer_scatter_indexing_map = affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %sum = arith.addi %idx0, %idx1 : index
      iree_linalg_ext.yield %sum, %idx1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x6xf32> -> tensor<2x6xf32>
  return %0 : tensor<2x6xf32>
}
// CHECK-LABEL: @map_store_hinted_non_result_symbol_use
//   CHECK-NOT: iree_vector_ext.transfer_scatter
//       CHECK: iree_linalg_ext.map_store

// -----

func.func @map_store_hinted_unused_symbol(
    %input: tensor<2x3xf32>, %output: tensor<2x6xf32>
) -> tensor<2x6xf32> {
  %0 = iree_linalg_ext.map_store {transfer_scatter_indexing_map = affine_map<(d0, d1)[s0, s1] -> (d0, s0)>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %sum = arith.addi %idx0, %idx1 : index
      iree_linalg_ext.yield %idx0, %sum, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x6xf32> -> tensor<2x6xf32>
  return %0 : tensor<2x6xf32>
}
// CHECK-LABEL: @map_store_hinted_unused_symbol
//   CHECK-NOT: iree_vector_ext.transfer_scatter
//       CHECK: iree_linalg_ext.map_store

// -----

func.func @map_store_hinted_duplicate_symbol(
    %input: tensor<2x3xf32>, %output: tensor<6x6xf32>
) -> tensor<6x6xf32> {
  %0 = iree_linalg_ext.map_store {transfer_scatter_indexing_map = affine_map<(d0, d1)[s0] -> (s0, s0)>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %sum = arith.addi %idx0, %idx1 : index
      iree_linalg_ext.yield %sum, %sum, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<6x6xf32> -> tensor<6x6xf32>
  return %0 : tensor<6x6xf32>
}
// CHECK-LABEL: @map_store_hinted_duplicate_symbol
//   CHECK-NOT: iree_vector_ext.transfer_scatter
//       CHECK: iree_linalg_ext.map_store

// -----

func.func @map_store_hinted_uncloneable_offset(
    %input: tensor<2x3xf32>, %output: tensor<2x3xf32>, %cond: i1
) -> tensor<2x3xf32> {
  %0 = iree_linalg_ext.map_store {transfer_scatter_indexing_map = affine_map<(d0, d1) -> (d0, d1)>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      // Identical branches deliberately introduce an uncloneable region op in
      // the offset slice.
      %out1 = scf.if %cond -> (index) {
        scf.yield %idx1 : index
      } else {
        scf.yield %idx1 : index
      }
      iree_linalg_ext.yield %idx0, %out1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
// CHECK-LABEL: @map_store_hinted_uncloneable_offset
//   CHECK-NOT: iree_vector_ext.transfer_scatter
//       CHECK: iree_linalg_ext.map_store
