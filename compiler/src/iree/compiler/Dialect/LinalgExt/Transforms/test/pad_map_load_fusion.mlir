// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(iree-linalg-ext-test-pad-map-load-fusion)" %s | FileCheck %s

// Basic: pad with zero low/high feeding an identity map_load.
// The pad should be folded into the map_load, with the padding value replacing
// the poison, and source indices adjusted by subtracting low_pad (all zero here).
func.func @fold_pad_into_identity_map_load(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %cst = arith.constant 0.0 : f32
  %padded = tensor.pad %arg0 low[0, 0] high[0, 0] {
  ^bb0(%b0: index, %b1: index):
    tensor.yield %cst : f32
  } : tensor<4x8xf32> to tensor<4x8xf32>
  %output = tensor.empty() : tensor<4x8xf32>
  %result = iree_linalg_ext.map_load %padded into %output {
  ^bb0(%idx0: index, %idx1: index):
    %poison = ub.poison : f32
    iree_linalg_ext.yield %idx0, %idx1, %poison : index, index, f32
  } : tensor<4x8xf32> into tensor<4x8xf32> -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}
// The subi with zero offset gets folded away, leaving identity indices.
// CHECK-LABEL: func.func @fold_pad_into_identity_map_load
//  CHECK-SAME:     %[[SRC:.+]]: tensor<4x8xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.{{0+}}e+00 : f32
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<4x8xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.map_load %[[SRC]] into %[[EMPTY]]
//       CHECK:     ^bb0(%[[I0:.+]]: index, %[[I1:.+]]: index):
//       CHECK:       iree_linalg_ext.yield %[[I0]], %[[I1]], %[[CST]]
//       CHECK:   return %[[RESULT]]

// -----

// Non-zero low padding: verify arith.subi with correct offsets.
func.func @fold_pad_nonzero_low(%arg0: tensor<4x8xf32>) -> tensor<5x10xf32> {
  %cst = arith.constant 0.0 : f32
  %padded = tensor.pad %arg0 low[1, 2] high[0, 0] {
  ^bb0(%b0: index, %b1: index):
    tensor.yield %cst : f32
  } : tensor<4x8xf32> to tensor<5x10xf32>
  %output = tensor.empty() : tensor<5x10xf32>
  %result = iree_linalg_ext.map_load %padded into %output {
  ^bb0(%idx0: index, %idx1: index):
    %poison = ub.poison : f32
    iree_linalg_ext.yield %idx0, %idx1, %poison : index, index, f32
  } : tensor<5x10xf32> into tensor<5x10xf32> -> tensor<5x10xf32>
  return %result : tensor<5x10xf32>
}
// CHECK-LABEL: func.func @fold_pad_nonzero_low
//  CHECK-SAME:     %[[SRC:.+]]: tensor<4x8xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.{{0+}}e+00 : f32
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<5x10xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.map_load %[[SRC]] into %[[EMPTY]]
//       CHECK:     ^bb0(%[[I0:.+]]: index, %[[I1:.+]]: index):
//       CHECK:       %[[ADJ0:.+]] = arith.subi %[[I0]], %[[C1]]
//       CHECK:       %[[ADJ1:.+]] = arith.subi %[[I1]], %[[C2]]
//       CHECK:       iree_linalg_ext.yield %[[ADJ0]], %[[ADJ1]], %[[CST]]
//       CHECK:   return %[[RESULT]]

// -----

// Pad with non-trivial map_load that already has index transforms (reversal).
// The subi adjustments should be appended after existing transforms.
func.func @fold_pad_into_nontrivial_map_load(%arg0: tensor<4x8xf32>) -> tensor<5x10xf32> {
  %cst = arith.constant 0.0 : f32
  %c3 = arith.constant 3 : index
  %c7 = arith.constant 7 : index
  %padded = tensor.pad %arg0 low[1, 2] high[0, 0] {
  ^bb0(%b0: index, %b1: index):
    tensor.yield %cst : f32
  } : tensor<4x8xf32> to tensor<5x10xf32>
  %output = tensor.empty() : tensor<5x10xf32>
  // The map_load reverses the indices before reading from source.
  %result = iree_linalg_ext.map_load %padded into %output {
  ^bb0(%idx0: index, %idx1: index):
    %rev0 = arith.subi %c3, %idx0 : index
    %rev1 = arith.subi %c7, %idx1 : index
    %poison = ub.poison : f32
    iree_linalg_ext.yield %rev0, %rev1, %poison : index, index, f32
  } : tensor<5x10xf32> into tensor<5x10xf32> -> tensor<5x10xf32>
  return %result : tensor<5x10xf32>
}
// CHECK-LABEL: func.func @fold_pad_into_nontrivial_map_load
//  CHECK-SAME:     %[[SRC:.+]]: tensor<4x8xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.{{0+}}e+00 : f32
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<5x10xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.map_load %[[SRC]] into %[[EMPTY]]
//       CHECK:     ^bb0(%[[I0:.+]]: index, %[[I1:.+]]: index):
//       CHECK:       %[[REV0:.+]] = arith.subi %[[C3]], %[[I0]]
//       CHECK:       %[[REV1:.+]] = arith.subi %[[C7]], %[[I1]]
//       CHECK:       %[[ADJ0:.+]] = arith.subi %[[REV0]], %[[C1]]
//       CHECK:       %[[ADJ1:.+]] = arith.subi %[[REV1]], %[[C2]]
//       CHECK:       iree_linalg_ext.yield %[[ADJ0]], %[[ADJ1]], %[[CST]]
//       CHECK:   return %[[RESULT]]

// -----

// Negative test: non-poison padding value — pattern should NOT match.
func.func @no_fold_non_poison_padding(%arg0: tensor<4x8xf32>) -> tensor<5x10xf32> {
  %cst = arith.constant 0.0 : f32
  %cst2 = arith.constant 1.0 : f32
  %padded = tensor.pad %arg0 low[1, 2] high[0, 0] {
  ^bb0(%b0: index, %b1: index):
    tensor.yield %cst : f32
  } : tensor<4x8xf32> to tensor<5x10xf32>
  %output = tensor.empty() : tensor<5x10xf32>
  // map_load already has a real padding value (not poison) — should not fold.
  %result = iree_linalg_ext.map_load %padded into %output {
  ^bb0(%idx0: index, %idx1: index):
    iree_linalg_ext.yield %idx0, %idx1, %cst2 : index, index, f32
  } : tensor<5x10xf32> into tensor<5x10xf32> -> tensor<5x10xf32>
  return %result : tensor<5x10xf32>
}
// CHECK-LABEL: func.func @no_fold_non_poison_padding
//       CHECK:   tensor.pad
//       CHECK:   iree_linalg_ext.map_load

// -----

// Negative test: non-constant pad value (depends on pad block args) —
// pattern should NOT match.
func.func @no_fold_non_constant_pad(%arg0: tensor<4x8xf32>) -> tensor<5x10xf32> {
  %padded = tensor.pad %arg0 low[1, 2] high[0, 0] {
  ^bb0(%b0: index, %b1: index):
    %idx_cast = arith.index_cast %b0 : index to i32
    %val = arith.sitofp %idx_cast : i32 to f32
    tensor.yield %val : f32
  } : tensor<4x8xf32> to tensor<5x10xf32>
  %output = tensor.empty() : tensor<5x10xf32>
  %result = iree_linalg_ext.map_load %padded into %output {
  ^bb0(%idx0: index, %idx1: index):
    %poison = ub.poison : f32
    iree_linalg_ext.yield %idx0, %idx1, %poison : index, index, f32
  } : tensor<5x10xf32> into tensor<5x10xf32> -> tensor<5x10xf32>
  return %result : tensor<5x10xf32>
}
// CHECK-LABEL: func.func @no_fold_non_constant_pad
//       CHECK:   tensor.pad
//       CHECK:   iree_linalg_ext.map_load %{{.+}} into

// -----

// Pad with both low and high padding and non-zero pad value.
func.func @fold_pad_low_and_high(%arg0: tensor<4x8xf32>) -> tensor<6x12xf32> {
  %cst = arith.constant 42.0 : f32
  %padded = tensor.pad %arg0 low[1, 2] high[1, 2] {
  ^bb0(%b0: index, %b1: index):
    tensor.yield %cst : f32
  } : tensor<4x8xf32> to tensor<6x12xf32>
  %output = tensor.empty() : tensor<6x12xf32>
  %result = iree_linalg_ext.map_load %padded into %output {
  ^bb0(%idx0: index, %idx1: index):
    %poison = ub.poison : f32
    iree_linalg_ext.yield %idx0, %idx1, %poison : index, index, f32
  } : tensor<6x12xf32> into tensor<6x12xf32> -> tensor<6x12xf32>
  return %result : tensor<6x12xf32>
}
// CHECK-LABEL: func.func @fold_pad_low_and_high
//  CHECK-SAME:     %[[SRC:.+]]: tensor<4x8xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 4.2{{0+}}e+01 : f32
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<6x12xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.map_load %[[SRC]] into %[[EMPTY]]
//       CHECK:     ^bb0(%[[I0:.+]]: index, %[[I1:.+]]: index):
//       CHECK:       %[[ADJ0:.+]] = arith.subi %[[I0]], %[[C1]]
//       CHECK:       %[[ADJ1:.+]] = arith.subi %[[I1]], %[[C2]]
//       CHECK:       iree_linalg_ext.yield %[[ADJ0]], %[[ADJ1]], %[[CST]]
//       CHECK:   return %[[RESULT]]
