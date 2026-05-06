// RUN: iree-opt --split-input-file --iree-util-test-affine-seed-dependency-analysis --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @seed_and_independent_args
util.func @seed_and_independent_args(%arg0: index, %arg1: index) {
  // CHECK: affine_seed_dependency = "seed0 = 1, seed1 = 0"
  "iree_unregistered.test_affine_seed_dependency"(%arg0, %arg0, %arg1) : (index, index, index) -> ()
  // CHECK: affine_seed_dependency = "seed0 = 0"
  "iree_unregistered.test_affine_seed_dependency"(%arg1, %arg0) : (index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @affine_apply_coefficients
util.func @affine_apply_coefficients(%arg0: index, %arg1: index) {
  %0 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4 + 7)>(%arg0, %arg1)
  // CHECK: affine_seed_dependency = "seed0 = 1, seed1 = 4"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0, %arg1) : (index, index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @arith_coefficients
util.func @arith_coefficients(%arg0: index, %arg1: index) {
  %c3 = arith.constant 3 : index
  %0 = arith.muli %arg1, %c3 : index
  %1 = arith.addi %arg0, %0 : index
  %2 = arith.subi %1, %arg1 : index
  // CHECK: affine_seed_dependency = "seed0 = 1, seed1 = 2"
  "iree_unregistered.test_affine_seed_dependency"(%2, %arg0, %arg1) : (index, index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @linearize_index_static_basis
util.func @linearize_index_static_basis(%arg0: index, %arg1: index) {
  %0 = affine.linearize_index [%arg0, %arg1] by (4, 8) : index
  // CHECK: affine_seed_dependency = "seed0 = 8, seed1 = 1"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0, %arg1) : (index, index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @delinearize_index_blocks_coefficients
util.func @delinearize_index_blocks_coefficients(%arg0: index) {
  %0:2 = affine.delinearize_index %arg0 into (4, 8) : index, index
  // CHECK: affine_seed_dependency = "seed0 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%0#1, %arg0) : (index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @non_linear_mul_is_per_seed_unknown
util.func @non_linear_mul_is_per_seed_unknown(%arg0: index, %arg1: index) {
  %0 = arith.muli %arg0, %arg1 : index
  // CHECK: affine_seed_dependency = "seed0 = ?, seed1 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0, %arg1) : (index, index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @coefficient_overflow_is_per_seed_unknown
util.func @coefficient_overflow_is_per_seed_unknown(%arg0: index) {
  %c2 = arith.constant 2 : index
  %cmax = arith.constant 9223372036854775807 : index
  %0 = arith.muli %arg0, %cmax : index
  %1 = arith.muli %0, %c2 : index
  // CHECK: affine_seed_dependency = "seed0 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%1, %arg0) : (index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @non_linear_affine_expr_is_per_seed_unknown
util.func @non_linear_affine_expr_is_per_seed_unknown(%arg0: index) {
  %0 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg0)
  // CHECK: affine_seed_dependency = "seed0 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0) : (index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @select_joins_coefficients
util.func @select_joins_coefficients(%arg0: index, %arg1: index, %cond: i1) {
  %0 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg0)
  %1 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg0)
  %2 = arith.select %cond, %0, %1 : index
  // CHECK: affine_seed_dependency = "seed0 = 1, seed1 = 0"
  "iree_unregistered.test_affine_seed_dependency"(%2, %arg0, %arg1) : (index, index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @select_conflicting_coefficients
util.func @select_conflicting_coefficients(%arg0: index, %cond: i1) {
  %0 = affine.apply affine_map<(d0) -> (d0)>(%arg0)
  %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
  %2 = arith.select %cond, %0, %1 : index
  // CHECK: affine_seed_dependency = "seed0 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%2, %arg0) : (index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @unsupported_op_is_full_unknown
util.func @unsupported_op_is_full_unknown(%arg0: index) {
  %0 = "iree_unregistered.unsupported_index_transform"(%arg0) : (index) -> index
  // CHECK: affine_seed_dependency = "unknown"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0) : (index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @map_store_affine_apply_constant_offset_on_seed1
func.func @map_store_affine_apply_constant_offset_on_seed1(
    %input: tensor<2x2xf32>, %output: tensor<2x4xf32>
) -> tensor<2x4xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 + 2)>(%idx1)
      // CHECK: affine_seed_dependency = "seed0 = 0, seed1 = 1"
      "iree_unregistered.test_affine_seed_dependency"(%1, %idx0, %idx1)
          : (index, index, index) -> ()
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x2xf32> into tensor<2x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: @map_store_linearize_collapse
func.func @map_store_linearize_collapse(
    %input: tensor<4x4xi32>, %output: tensor<16xi32>
) -> tensor<16xi32> {
  %mask = arith.constant true
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%arg2: index, %arg3: index):
      %2 = affine.linearize_index disjoint [%arg2, %arg3] by (4, 4) : index
      // CHECK: affine_seed_dependency = "seed0 = 4, seed1 = 1"
      "iree_unregistered.test_affine_seed_dependency"(%2, %arg2, %arg3)
          : (index, index, index) -> ()
      iree_linalg_ext.yield %2, %mask : index, i1
  } : tensor<4x4xi32> into tensor<16xi32> -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// CHECK-LABEL: @map_store_delinearize_expand
func.func @map_store_delinearize_expand(
    %input: tensor<16xf32>, %output: tensor<4x4xf32>
) -> tensor<4x4xf32> {
  %mask = arith.constant true
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%arg2: index):
      %2:2 = affine.delinearize_index %arg2 into (4, 4) : index, index
      "iree_unregistered.test_affine_seed_dependency"(%2#0, %arg2)
          : (index, index) -> ()
      "iree_unregistered.test_affine_seed_dependency"(%2#1, %arg2)
          : (index, index) -> ()
      iree_linalg_ext.yield %2#0, %2#1, %mask : index, index, i1
  } : tensor<16xf32> into tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// CHECK-COUNT-2: affine_seed_dependency = "seed0 = ?"

// -----

// CHECK-LABEL: @map_store_nested_gpu_delinearize
func.func @map_store_nested_gpu_delinearize(
    %input: tensor<256x256xf32>,
    %output: tensor<2x16x8x4x4x4x4xf32>
) -> tensor<2x16x8x4x4x4x4xf32> {
  %mask = arith.constant true
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%arg0: index, %arg1: index):
      %13:2 = affine.delinearize_index %arg0 into (2, 128) : index, index
      %14:2 = affine.delinearize_index %arg1 into (16, 16) : index, index
      %15:3 = affine.delinearize_index %13#1
          into (4, 8, 4) : index, index, index
      %16:2 = affine.delinearize_index %14#1 into (4, 4) : index, index
      "iree_unregistered.test_affine_seed_dependency"(%13#0, %arg0, %arg1)
          : (index, index, index) -> ()
      "iree_unregistered.test_affine_seed_dependency"(%13#1, %arg0, %arg1)
          : (index, index, index) -> ()
      "iree_unregistered.test_affine_seed_dependency"(%15#2, %arg0, %arg1)
          : (index, index, index) -> ()
      "iree_unregistered.test_affine_seed_dependency"(%16#1, %arg0, %arg1)
          : (index, index, index) -> ()
      iree_linalg_ext.yield %13#0, %14#0, %15#1, %16#1, %15#0, %15#2,
          %16#0, %mask : index, index, index, index, index, index, index, i1
  } : tensor<256x256xf32> into tensor<2x16x8x4x4x4x4xf32>
      -> tensor<2x16x8x4x4x4x4xf32>
  return %0 : tensor<2x16x8x4x4x4x4xf32>
}
// CHECK-COUNT-3: affine_seed_dependency = "seed0 = ?, seed1 = 0"
// CHECK: affine_seed_dependency = "seed0 = 0, seed1 = ?"
