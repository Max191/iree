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
