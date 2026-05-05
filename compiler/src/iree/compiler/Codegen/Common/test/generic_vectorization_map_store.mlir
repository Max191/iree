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
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C0]]]
//  CHECK-SAME:     vector<2xindex>
//  CHECK-SAME:     : vector<2x2xf32>, tensor<2x4xf32> -> tensor<2x4xf32>
//       CHECK:   return %[[SCATTER]] : tensor<2x4xf32>

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
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C0]]]
//  CHECK-SAME:     vector<2x3xindex>
//  CHECK-SAME:     : vector<2x3xf32>, tensor<2x5xf32> -> tensor<2x5xf32>
//       CHECK:   return %[[SCATTER]] : tensor<2x5xf32>

// -----

// Full-byte map_store still vectorizes when the mask depends on the inner dim.
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
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[MASK:.+]] = arith.cmpi uge
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_vector_ext.transfer_scatter %[[READ]] into %[[OUTPUT]][%[[C0]], %[[C0]]], %[[MASK]]
//  CHECK-SAME:     : vector<2x2xf32>, tensor<2x2xf32>, vector<2xi1> -> tensor<2x2xf32>
//       CHECK:   return %[[SCATTER]] : tensor<2x2xf32>

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
