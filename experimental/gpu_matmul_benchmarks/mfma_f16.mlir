func.func @foo(%lhs: tensor<?x?x8x4x16x2x4xf16>, %rhs: tensor<?x?x4x2x4x16x2x4xf16>) -> tensor<?x?x8x4x2x4x16x4xf32> {
  %c0 = arith.constant 0 : index
  %c0_f32 = arith.constant 0.0 : f32
  %m = tensor.dim %lhs, %c0 : tensor<?x?x8x4x16x2x4xf16>
  %n = tensor.dim %rhs, %c0 : tensor<?x?x4x2x4x16x2x4xf16>
  %empty = tensor.empty(%m, %n) : tensor<?x?x8x4x2x4x16x4xf32>
  %zero = linalg.fill ins(%c0_f32 : f32) outs(%empty : tensor<?x?x8x4x2x4x16x4xf32>) -> tensor<?x?x8x4x2x4x16x4xf32>
  %result = iree_gpu.multi_mma %lhs, %rhs, %zero {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ], iterator_types = [
      #iree_gpu.iterator_type<parallel>,
      #iree_gpu.iterator_type<parallel>,
      #iree_gpu.iterator_type<reduction>
    ], kind = #iree_gpu.data_tiled_mma_layout<
      intrinsic =  MFMA_F32_16x16x16_F16,
      unroll_m = 8,
      unroll_n = 2,
      unroll_n_to_subgroups = 4,
      unroll_k = 2
    >} : tensor<?x?x8x4x16x2x4xf16>, tensor<?x?x4x2x4x16x2x4xf16> into tensor<?x?x8x4x2x4x16x4xf32>
  return %result : tensor<?x?x8x4x2x4x16x4xf32>
}
