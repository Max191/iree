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
  // CHECK-SAME: affine_seed_expression = "d0 + d1 * 4 + 7"
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

// CHECK-LABEL: @scale_by_zero_is_independent
util.func @scale_by_zero_is_independent(%arg0: index, %arg1: index) {
  %c0 = arith.constant 0 : index
  %0 = arith.muli %arg0, %c0 : index
  // CHECK: affine_seed_dependency = "seed0 = 0, seed1 = 0"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0, %arg1) : (index, index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @cfg_join_conflicting_coefficients
func.func @cfg_join_conflicting_coefficients(%arg0: index, %cond: i1) {
  %c2 = arith.constant 2 : index
  %0 = arith.muli %arg0, %c2 : index
  cf.cond_br %cond, ^bb1(%arg0 : index), ^bb1(%0 : index)
^bb1(%merged: index):
  // CHECK: affine_seed_dependency = "seed0 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%merged, %arg0) : (index, index) -> ()
  return
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
  // CHECK-SAME: affine_seed_expression = "d0 floordiv 2"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0) : (index, index) -> ()
  %1 = affine.apply affine_map<(d0) -> (d0 ceildiv 4)>(%arg0)
  // CHECK: affine_seed_dependency = "seed0 = ?"
  // CHECK-SAME: affine_seed_expression = "d0 ceildiv 4"
  "iree_unregistered.test_affine_seed_dependency"(%1, %arg0) : (index, index) -> ()
  %2 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg0)
  // CHECK: affine_seed_dependency = "seed0 = ?"
  // CHECK-SAME: affine_seed_expression = "d0 mod 8"
  "iree_unregistered.test_affine_seed_dependency"(%2, %arg0) : (index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @non_linear_affine_expr_with_known_offset
util.func @non_linear_affine_expr_with_known_offset(%arg0: index) {
  %0 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg0)
  // CHECK: affine_seed_dependency = "seed0 = ?"
  // CHECK-SAME: affine_seed_expression = "(d0 + 1) floordiv 2"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0) : (index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @non_separable_dynamic_offset_invalidates_seed
util.func @non_separable_dynamic_offset_invalidates_seed(%arg0: index, %arg1: index) {
  %0 = affine.apply affine_map<(d0, d1) -> ((d0 + d1) floordiv 2)>(%arg0, %arg1)
  // CHECK: affine_seed_dependency = "seed0 = ?"
  // CHECK-SAME: affine_seed_expression = "0 invalidated"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0) : (index, index) -> ()
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

// CHECK-LABEL: @unsupported_op_with_independent_operand_is_full_unknown
util.func @unsupported_op_with_independent_operand_is_full_unknown(%arg0: index) {
  %c0 = arith.constant 0 : index
  %0 = "iree_unregistered.unsupported_index_transform"(%c0) : (index) -> index
  // CHECK: affine_seed_dependency = "unknown"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0) : (index, index) -> ()
  util.return
}
