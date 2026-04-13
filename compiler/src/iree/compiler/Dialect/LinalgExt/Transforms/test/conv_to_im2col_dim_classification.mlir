// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-linalg-ext-convert-conv-to-im2col-op))" %s | FileCheck %s

util.func public @conv_2d_nhwc_hwcf_phase1(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
     ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%arg2 : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}

// CHECK-LABEL: util.func public @conv_2d_nhwc_hwcf_phase1
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x14x14x36xf32>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [1, 1, 1] dilations = [1, 1, 1] kernel_size = [1, 3, 3]
// CHECK-SAME:   offsets = [0, 0, 0, 0] output_sizes =
// CHECK-SAME:   [1], [14], [14], [1, 3, 3, 4]
// CHECK-SAME:   batch_pos = [] m_pos = [0, 1, 2] k_pos = [3]
// CHECK-SAME:   input_k_perm = [0, 1, 2, 3] output_perm = [0, 1, 2, 3]
// CHECK-SAME:   ins(%arg0 : tensor<1x16x16x4xf32>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<1x14x14x36xf32>) -> tensor<1x14x14x36xf32>
// CHECK:      %[[FILTER:.+]] = tensor.collapse_shape %arg1
// CHECK-SAME:   into tensor<36x16xf32>
// CHECK:      linalg.generic
// CHECK-SAME:   ins(%[[IM2COL]], %[[FILTER]] : tensor<1x14x14x36xf32>, tensor<36x16xf32>)

// -----

util.func public @conv_2d_nhwgc_gfhwc_phase1(%arg0: tensor<2x10x10x7x4xf32>, %arg1: tensor<7x16x3x3x4xf32>, %arg2: tensor<2x8x8x7x16xf32>) -> tensor<2x8x8x7x16xf32> {
  %0 = linalg.conv_2d_nhwgc_gfhwc
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
     ins(%arg0, %arg1 : tensor<2x10x10x7x4xf32>, tensor<7x16x3x3x4xf32>)
    outs(%arg2 : tensor<2x8x8x7x16xf32>) -> tensor<2x8x8x7x16xf32>
  util.return %0 : tensor<2x8x8x7x16xf32>
}

// CHECK-LABEL: util.func public @conv_2d_nhwgc_gfhwc_phase1
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<7x2x8x8x36xf32>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [1, 1, 1] dilations = [1, 1, 1] kernel_size = [1, 3, 3]
// CHECK-SAME:   offsets = [0, 0, 0, 0, 0] output_sizes =
// CHECK-SAME:   [7], [2], [8], [8], [1, 3, 3, 4]
// CHECK-SAME:   batch_pos = [3] m_pos = [0, 1, 2] k_pos = [4]
// CHECK-SAME:   input_k_perm = [0, 1, 2, 3] output_perm = [0, 1, 2, 3, 4]
// CHECK-SAME:   ins(%arg0 : tensor<2x10x10x7x4xf32>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<7x2x8x8x36xf32>) -> tensor<7x2x8x8x36xf32>
// CHECK:      %[[FILTER:.+]] = tensor.collapse_shape %arg1
// CHECK-SAME:   into tensor<7x16x36xf32>
// CHECK:      linalg.generic
// CHECK-SAME:   ins(%[[IM2COL]], %[[FILTER]] : tensor<7x2x8x8x36xf32>, tensor<7x16x36xf32>)
