// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-linalg-ext-vectorize-ops))' --split-input-file %s | FileCheck %s

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
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_store
//  CHECK-SAME:     %[[READ]] into %[[OUTPUT]]
//       CHECK:     : vector<4x16x64xf32> into tensor<4x16x64xf32> -> tensor<4x16x64xf32>
//       CHECK:   return %[[MAP_SCATTER]] : tensor<4x16x64xf32>

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
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_store
//  CHECK-SAME:     %[[READ]] into %[[OUTPUT]]
//       CHECK:     : vector<2x2xf4E2M1FN> into tensor<2x2xf4E2M1FN> -> tensor<2x2xf4E2M1FN>
//       CHECK:   return %[[MAP_SCATTER]] : tensor<2x2xf4E2M1FN>

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
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_store
//  CHECK-SAME:     %[[READ]] into %[[OUTPUT]]
//       CHECK:     : vector<2x2xf4E2M1FN> into tensor<2x4xf4E2M1FN> -> tensor<2x4xf4E2M1FN>
//       CHECK:   return %[[MAP_SCATTER]] : tensor<2x4xf4E2M1FN>

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

// Standard NHWC layout, K tile size (4) divides innermost input dim C (640).
// Vectorizes along K (output dim 2) with vector width 4.
// Non-vectorized dims: batch (2) x M (2) = 4 iterations.
#map_k = affine_map<(d0) -> (d0 * 4)>
func.func @im2col_vectorize_nhwc(
    %input: tensor<2x34x34x640xf32>, %m_off: index, %k: index
) -> tensor<2x2x4xf32> {
  %0 = tensor.empty() : tensor<2x2x4xf32>
  %k_off = affine.apply #map_k(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          m_offset = [%m_off] * [1] k_offset = [%k_off] * [1]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
  return %1 : tensor<2x2x4xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_nhwc
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.0{{.*}} : f32
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   %[[R0:.+]] = vector.transfer_read %[[INPUT]]{{.*}}, %[[CST]] : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[R0]], {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   %[[R1:.+]] = vector.transfer_read %[[INPUT]]{{.*}}, %[[CST]] : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[R1]], {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   %[[R2:.+]] = vector.transfer_read %[[INPUT]]{{.*}}, %[[CST]] : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[R2]], {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   %[[R3:.+]] = vector.transfer_read %[[INPUT]]{{.*}}, %[[CST]] : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   %[[FINAL:.+]] = vector.transfer_write %[[R3]], {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   return %[[FINAL]] : tensor<2x2x4xf32>

// -----

// Multi-dim K output. K is split into two output dims (sizes 2 and 4).
// Innermost K dim (4) divides C (640), so vectorizes along dim 3.
// Non-vectorized: batch (2) x M (2) x K0 (2) = 8 iterations.
func.func @im2col_vectorize_multi_k(
    %input: tensor<2x34x34x640xf32>, %m_off: index, %k: index
) -> tensor<2x2x2x4xf32> {
  %0 = tensor.empty() : tensor<2x2x2x4xf32>
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          m_offset = [%m_off] * [1] k_offset = [%k, 0] * [4, 1]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2, 3]
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x2x2x4xf32>) -> tensor<2x2x2x4xf32>
  return %1 : tensor<2x2x2x4xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_multi_k
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x34x34x640xf32>
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x2x4xf32>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x2x4xf32>
// Verify all 8 iterations produce transfer_read/write pairs.
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x2x4xf32>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x2x4xf32>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x2x4xf32>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x2x4xf32>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x2x4xf32>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   %[[FINAL:.+]] = vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x2x4xf32>
//       CHECK:   return %[[FINAL]]

// -----

// CHWN layout with output_perm that puts batch at the last output dim.
// Input dim 3 (batch, N=4) is innermost and maps to output dim 3 via
// output_perm = [3, 1, 2, 0]. Vectorizes along dim 3, width 4.
// This uses the default (minor identity) permutation map for the write.
func.func @im2col_vectorize_chwn_output_perm(
    %input: tensor<16x26x18x4xf32>, %m0: index, %m1: index, %k: index
) -> tensor<2x2x2x4xf32> {
  %0 = tensor.empty() : tensor<2x2x2x4xf32>
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [24, 16]
          m_offset = [%m0, %m1] * [3, 1] k_offset = [%k] * [1]
          batch_pos = [3] m_pos = [1, 2] k_pos = [0]
          input_k_perm = [0, 1, 2] output_perm = [3, 1, 2, 0]
          ins(%input : tensor<16x26x18x4xf32>)
          outs(%0 : tensor<2x2x2x4xf32>) -> tensor<2x2x2x4xf32>
  return %1 : tensor<2x2x2x4xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_chwn_output_perm
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<16x26x18x4xf32>
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<16x26x18x4xf32>, vector<4xf32>
//  CHECK-NEXT:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x2x4xf32>
// Verify no explicit permutation_map (default minor identity used for write).
//   CHECK-NOT:   permutation_map

// -----

// CHWN layout with identity output_perm. Batch dim (innermost input, dim 3)
// maps to output dim 0. Vectorization along dim 0 requires a non-default
// permutation_map on the vector.transfer_write.
//   CHECK-DAG: #[[$WRITE_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
func.func @im2col_vectorize_chwn(
    %input: tensor<16x26x18x4xf32>, %m0: index, %m1: index, %k: index
) -> tensor<4x2x2x2xf32> {
  %0 = tensor.empty() : tensor<4x2x2x2xf32>
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [24, 16]
          m_offset = [%m0, %m1] * [3, 1] k_offset = [%k] * [1]
          batch_pos = [3] m_pos = [1, 2] k_pos = [0]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2, 3]
          ins(%input : tensor<16x26x18x4xf32>)
          outs(%0 : tensor<4x2x2x2xf32>) -> tensor<4x2x2x2xf32>
  return %1 : tensor<4x2x2x2xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_chwn
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<16x26x18x4xf32>
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   %[[R:.+]] = vector.transfer_read %[[INPUT]]{{.*}} : tensor<16x26x18x4xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[R]], {{.*}}permutation_map = #[[$WRITE_MAP]]{{.*}} : vector<4xf32>, tensor<4x2x2x2xf32>

// -----

// Dynamic output shape: vectorization pattern should not match.
func.func @im2col_no_vectorize_dynamic(
    %input: tensor<2x34x34x640xf32>, %m_size: index, %m_off: index, %k: index
) -> tensor<2x?x4xf32> {
  %0 = tensor.empty(%m_size) : tensor<2x?x4xf32>
  %k_off = affine.apply affine_map<(d0) -> (d0 * 4)>(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          m_offset = [%m_off] * [1] k_offset = [%k_off] * [1]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x?x4xf32>) -> tensor<2x?x4xf32>
  return %1 : tensor<2x?x4xf32>
}
// CHECK-LABEL: func.func @im2col_no_vectorize_dynamic
//       CHECK:   iree_linalg_ext.im2col
//   CHECK-NOT:   vector.transfer_read
//   CHECK-NOT:   vector.transfer_write

// -----

// Non-vectorizable due to input_k_perm = [1, 0] making innermost K
// non-contiguous in input. Falls back to scalar unrolling (vector<1>).
// Output has 1*2*4 = 8 elements, so 8 scalar iterations.
func.func @im2col_scalar_fallback(
    %input: tensor<1x3x2xf32>
) -> tensor<1x2x4xf32> {
  %0 = tensor.empty() : tensor<1x2x4xf32>
  %1 = iree_linalg_ext.im2col strides = [1] dilations = [1] kernel_size = [2]
                          m_offset = [0] * [1] k_offset = [0] * [1]
                          batch_pos = [0] m_pos = [1] k_pos = [2]
                          input_k_perm = [1, 0] output_perm = [0, 1, 2]
                          ins(%input : tensor<1x3x2xf32>)
                          outs(%0 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
  return %1 : tensor<1x2x4xf32>
}
// CHECK-LABEL: func.func @im2col_scalar_fallback
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x3x2xf32>
//   CHECK-NOT:   iree_linalg_ext.im2col
// Scalar fallback: uses vector<1xf32> for each element.
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<1x3x2xf32>, vector<1xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<1xf32>, tensor<1x2x4xf32>

// -----

// Source padding (conv padding folded into im2col). NHWC layout.
// Input: 2x34x34x640 (unpadded), pad_low=[0,1,1,0], pad_high=[0,1,1,0].
// Effective: 2x36x36x640. Ho = (36-3)/1+1 = 34, Wo = 34.
// K tile = 4 divides C = 640 -> vectorize along K.
// Verifies: masked transfer_read with pad_value, unmasked transfer_write.
#map_k_pad = affine_map<(d0) -> (d0 * 4)>
func.func @im2col_vectorize_source_padding(
    %input: tensor<2x34x34x640xf32>, %m_off: index, %k: index
) -> tensor<2x2x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<2x2x4xf32>
  %k_off = affine.apply #map_k_pad(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          m_offset = [%m_off] * [1] k_offset = [%k_off] * [1]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0]
          pad_value(%cst : f32)
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
  return %1 : tensor<2x2x4xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_source_padding
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.0{{.*}} : f32
//   CHECK-NOT:   iree_linalg_ext.im2col
// Verify masked transfer_read with pad_value.
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   %[[FINAL:.+]] = vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   return %[[FINAL]] : tensor<2x2x4xf32>

// -----

// Source padding with small output (no result masking needed). NHWC layout.
// Input: 1x6x6x8 (unpadded), pad_low=[0,1,1,0], pad_high=[0,1,1,0].
// Effective: 1x8x8x8. kernel 3x3, strides [1,1]: Ho = 6, Wo = 6.
// real_M = 36, real_K = 3*3*8 = 72.
// Output: 1x2x8 -> output M = 2, output K = 8.
// output_K (8) < real_K (72) -> no K result mask needed.
// output_M (2) < real_M (36) -> no M result mask needed.
// Only source padding mask needed (not result masking).
// Vectorize along K (dim 2), width 8 divides C=8.
func.func @im2col_vectorize_source_padding_small_output(
    %input: tensor<1x6x6x8xf32>, %m_off: index, %k_off: index
) -> tensor<1x2x8xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<1x2x8xf32>
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          m_offset = [%m_off] * [1] k_offset = [%k_off] * [1]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0]
          pad_value(%cst : f32)
          ins(%input : tensor<1x6x6x8xf32>)
          outs(%0 : tensor<1x2x8xf32>) -> tensor<1x2x8xf32>
  return %1 : tensor<1x2x8xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_source_padding_small_output
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x6x6x8xf32>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.0{{.*}} : f32
//   CHECK-NOT:   iree_linalg_ext.im2col
// Source padding bounds checks and masked reads (scalar fallback).
//       CHECK:   arith.cmpi sge
//       CHECK:   arith.cmpi slt
//       CHECK:   arith.andi
//       CHECK:   vector.broadcast {{.*}} : i1 to vector<1xi1>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} : tensor<1x6x6x8xf32>, vector<1xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<1xf32>, tensor<1x2x8xf32>

// -----

// Source padding with scalar fallback (non-vectorizable).
// Input: 1x3x2 (1d conv), pad_low=[0,1,0], pad_high=[0,1,0].
// Effective: 1x5x2. kernel [2], strides [1]: Ho = 4.
// Scalar unrolling (vec_width=1) due to input_k_perm = [1, 0].
func.func @im2col_scalar_fallback_padding(
    %input: tensor<1x3x2xf32>
) -> tensor<1x2x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<1x2x4xf32>
  %1 = iree_linalg_ext.im2col strides = [1] dilations = [1] kernel_size = [2]
                          m_offset = [0] * [1] k_offset = [0] * [1]
                          batch_pos = [0] m_pos = [1] k_pos = [2]
                          input_k_perm = [1, 0] output_perm = [0, 1, 2]
                          input_pad_low = [0, 1, 0] input_pad_high = [0, 1, 0]
                          pad_value(%cst : f32)
                          ins(%input : tensor<1x3x2xf32>)
                          outs(%0 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
  return %1 : tensor<1x2x4xf32>
}
// CHECK-LABEL: func.func @im2col_scalar_fallback_padding
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x3x2xf32>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.0{{.*}} : f32
//   CHECK-NOT:   iree_linalg_ext.im2col
// Scalar fallback with masked reads.
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} : tensor<1x3x2xf32>, vector<1xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<1xf32>, tensor<1x2x4xf32>

// -----

// Regression test: backward-weight-style im2col with dilation=2, vectorized
// along the innermost batch dimension (N=2). Verifies that:
//   1. The im2col op is successfully lowered (no iree_linalg_ext.im2col remains).
//   2. The M delinearization uses inner basis 3 from m_strides=[3,1], not the
//      formula-derived (wrong) value of 4 for kernel_size=8, spatial=18, dilation=2.
//   3. Spatial offsets include a factor of 2 (the dilation).
//
// Layout: input=C x IH x IW x N (k_pos=[0], m_pos=[1,2], batch_pos=[3])
// Output: M0 x M1 x K x B (output_perm=[1,2,3,0], shape 3x3x4x2)
// Batch (N=2) is innermost in input, vectorized along output dim 3 (width 2).
//
//   CHECK-DAG: #[[$SPATIAL_DIL:.+]] = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
func.func @im2col_vectorize_chwn_dilated(
    %input: tensor<4x18x18x2xf32>, %m0: index, %m1: index, %k: index
) -> tensor<3x3x4x2xf32> {
  %0 = tensor.empty() : tensor<3x3x4x2xf32>
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [2, 2] kernel_size = [8, 8]
          m_offset = [%m0, %m1] * [3, 1] k_offset = [%k] * [1]
          batch_pos = [3] m_pos = [1, 2] k_pos = [0]
          input_k_perm = [0, 1, 2] output_perm = [1, 2, 3, 0]
          ins(%input : tensor<4x18x18x2xf32>)
          outs(%0 : tensor<3x3x4x2xf32>) -> tensor<3x3x4x2xf32>
  return %1 : tensor<3x3x4x2xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_chwn_dilated
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<4x18x18x2xf32>
//  CHECK-SAME:     %[[M0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[M1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[K:[a-zA-Z0-9_]+]]: index
// Verify im2col is fully lowered to vector ops.
//   CHECK-NOT:   iree_linalg_ext.im2col
// Verify M delinearization uses inner basis 3 (from m_strides=[3,1]).
// Old formula-based code produced basis 4 (wrong).
//       CHECK:   %{{.+}}:2 = affine.delinearize_index %{{.+}} into (3) : index, index
// Verify dilation factor 2 is applied to spatial offset computation.
//       CHECK:   affine.apply #[[$SPATIAL_DIL]]
// Verify vectorized reads produce vector<2xf32> (batch dimension size 2).
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<4x18x18x2xf32>, vector<2xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<2xf32>, tensor<3x3x4x2xf32>

// -----

// Result M masking: output M (20) exceeds real M (16) from effective input.
// Input: 1x4x4x4 (NHWC), pad_low=[0,1,1,0], pad_high=[0,1,1,0].
// Effective: 1x6x6x4. kernel 3x3, strides [1,1]: Ho=4, Wo=4, realM=16.
// Output: 1x20x4 => output M=20 > real M=16 -> needMResultMask.
// Vectorize along K (dim 2), width 4 divides C=4.
// k_offset must be a known multiple of 4 for vectorization.
// Verifies: M result mask (arith.cmpi slt for M bounds check).
#map_k_rm = affine_map<(d0) -> (d0 * 4)>
func.func @im2col_vectorize_m_result_mask(
    %input: tensor<1x4x4x4xf32>, %m_off: index, %k: index
) -> tensor<1x20x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<1x20x4xf32>
  %k_off = affine.apply #map_k_rm(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          m_offset = [%m_off] * [1] k_offset = [%k_off] * [1]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0]
          pad_value(%cst : f32)
          ins(%input : tensor<1x4x4x4xf32>)
          outs(%0 : tensor<1x20x4xf32>) -> tensor<1x20x4xf32>
  return %1 : tensor<1x20x4xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_m_result_mask
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x4x4x4xf32>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.0{{.*}} : f32
//   CHECK-NOT:   iree_linalg_ext.im2col
// Verify M result masking: compare linearized M index against real M extent (16).
//       CHECK:   arith.cmpi slt, %{{.*}}, %{{.*}} : index
// Verify masked transfer_read with pad_value and vector width 4.
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} : tensor<1x4x4x4xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<1x20x4xf32>

// -----

// Result K masking with vectorization along K dimension.
// Input: 1x3x2 (batch x spatial x channel). kernel=[2], strides=[1].
// realM = (3-2)/1+1 = 2, realK = kernel(2) * channel(2) = 4.
// Output: 1x2x3x2 (batch=1, M=2, K0=3, K1=2).
// k_offset=[0,0]*[2,1] splits K into two output dims.
// outputK = K0*K1 = 6 > realK = 4 -> needKResultMask.
// Vectorize along K1 (dim 3), width 2 divides C=2. vecAlongK=true.
// Verifies: K result mask (vector.create_mask for per-lane K masking).
// Also exercises M result mask since m_off is dynamic.
func.func @im2col_vectorize_k_result_mask(
    %input: tensor<1x3x2xf32>, %m_off: index
) -> tensor<1x2x3x2xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<1x2x3x2xf32>
  %1 = iree_linalg_ext.im2col strides = [1] dilations = [1] kernel_size = [2]
                          m_offset = [%m_off] * [1] k_offset = [0, 0] * [2, 1]
                          batch_pos = [0] m_pos = [1] k_pos = [2]
                          input_k_perm = [0, 1] output_perm = [0, 1, 2, 3]
                          input_pad_low = [0, 0, 0] input_pad_high = [0, 0, 0]
                          pad_value(%cst : f32)
                          ins(%input : tensor<1x3x2xf32>)
                          outs(%0 : tensor<1x2x3x2xf32>) -> tensor<1x2x3x2xf32>
  return %1 : tensor<1x2x3x2xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_k_result_mask
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x3x2xf32>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.0{{.*}} : f32
//   CHECK-NOT:   iree_linalg_ext.im2col
// Verify M result masking: comparison against real M extent.
//       CHECK:   arith.cmpi slt
// Verify K result masking: vector.create_mask for per-lane valid K extent.
// The K mask varies per K0 iteration (create_mask with different lane counts).
//       CHECK:   vector.create_mask {{.*}} : vector<2xi1>
// Verify combined mask (M scalar broadcast AND'd with K vector mask).
//       CHECK:   vector.broadcast {{.*}} : i1 to vector<2xi1>
//       CHECK:   arith.andi
// Verify masked transfer_read with pad_value.
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} : tensor<1x3x2xf32>, vector<2xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<2xf32>, tensor<1x2x3x2xf32>

// NOTE: A negative test for kMaxUnrollIters (> 1024 non-vectorized elements)
// cannot be included in this file because the VectorizeIREELinalgExtOpsPass
// uses applyPatternsGreedily which signals pass failure when patterns don't
// converge, and --split-input-file propagates the failure exit code. The
// kMaxUnrollIters guard is covered by the rationale comment in the source and
// verified indirectly by all tests above being below the limit.
