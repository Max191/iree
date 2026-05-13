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

// CHECK-LABEL: @index_cast_passthrough
util.func @index_cast_passthrough(%arg0: index) {
  %0 = arith.index_cast %arg0 : index to i64
  // CHECK: affine_seed_dependency = "seed0 = 1"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0) : (i64, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @index_cast_wide_integer_passthrough
util.func @index_cast_wide_integer_passthrough(%arg0: index) {
  %0 = arith.index_cast %arg0 : index to i128
  // CHECK: affine_seed_dependency = "seed0 = 1"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0) : (i128, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @index_cast_narrowing_is_conservative
util.func @index_cast_narrowing_is_conservative(%arg0: index) {
  %0 = arith.index_cast %arg0 : index to i32
  // CHECK: affine_seed_dependency = "seed0 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0) : (i32, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @index_cast_to_index_is_conservative
util.func @index_cast_to_index_is_conservative(%arg0: i64) {
  %0 = arith.index_cast %arg0 : i64 to index
  // CHECK: affine_seed_dependency = "seed0 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0) : (index, i64) -> ()
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

// CHECK-LABEL: @linearize_index_dynamic_basis_is_conservative
util.func @linearize_index_dynamic_basis_is_conservative(%arg0: index, %arg1: index, %basis: index) {
  %0 = affine.linearize_index [%arg0, %arg1] by (4, %basis) : index
  // CHECK: affine_seed_dependency = "seed0 = ?, seed1 = ?, seed2 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0, %arg1, %basis) : (index, index, index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @linearize_index_stride_overflow_is_conservative
util.func @linearize_index_stride_overflow_is_conservative(%arg0: index, %arg1: index, %arg2: index) {
  %0 = affine.linearize_index [%arg0, %arg1, %arg2] by (3037000500, 3037000500, 3037000500) : index
  // CHECK: affine_seed_dependency = "seed0 = ?, seed1 = ?, seed2 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0, %arg1, %arg2) : (index, index, index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @delinearize_index_is_conservative
util.func @delinearize_index_is_conservative(%arg0: index) {
  %0:2 = affine.delinearize_index %arg0 into (4, 8) : index, index
  // CHECK: affine_seed_dependency = "seed0 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%0#0, %arg0) : (index, index) -> ()
  // CHECK: affine_seed_dependency = "seed0 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%0#1, %arg0) : (index, index) -> ()
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

// CHECK-LABEL: @select_matching_values
util.func @select_matching_values(%arg0: index, %cond: i1) {
  // Matching arms preserve the affine relationship when the condition is
  // independent of the seed.
  %0 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg0)
  %1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg0)
  %2 = arith.select %cond, %0, %1 : index
  // CHECK: affine_seed_dependency = "seed0 = 1"
  "iree_unregistered.test_affine_seed_dependency"(%2, %arg0) : (index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @select_conflicting_values_is_conservative
util.func @select_conflicting_values_is_conservative(%arg0: index, %cond: i1) {
  %0 = affine.apply affine_map<(d0) -> (d0)>(%arg0)
  %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
  %2 = arith.select %cond, %0, %1 : index
  // CHECK: affine_seed_dependency = "seed0 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%2, %arg0) : (index, index) -> ()
  util.return
}

// -----

// CHECK-LABEL: @select_seed_dependent_condition_is_conservative
util.func @select_seed_dependent_condition_is_conservative(%arg0: i1) {
  // The selected values are seed-independent, but the predicate itself is a
  // seed and may control the produced index value.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.select %arg0, %c0, %c1 : index
  // CHECK: affine_seed_dependency = "seed0 = ?"
  "iree_unregistered.test_affine_seed_dependency"(%0, %arg0) : (index, i1) -> ()
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

// CHECK-LABEL: @scf_for_iv_lower_bound_seed
func.func @scf_for_iv_lower_bound_seed(%lb: index) {
  %c1 = arith.constant 1 : index
  %ub = affine.apply affine_map<(d0) -> (d0 + 4)>(%lb)
  scf.for %iv = %lb to %ub step %c1 {
    // CHECK: affine_seed_dependency = "seed0 = 1"
    "iree_unregistered.test_affine_seed_dependency"(%iv, %lb) : (index, index) -> ()
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_iv_lower_bound_producer
func.func @scf_for_iv_lower_bound_producer(%arg0: index) {
  %c1 = arith.constant 1 : index
  %lb = affine.apply affine_map<(d0) -> (d0 * 2 + 5)>(%arg0)
  %ub = affine.apply affine_map<(d0) -> (d0 + 4)>(%lb)
  scf.for %iv = %lb to %ub step %c1 {
    // CHECK: affine_seed_dependency = "seed0 = 2"
    "iree_unregistered.test_affine_seed_dependency"(%iv, %arg0) : (index, index) -> ()
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_iv_upper_bound_seed
func.func @scf_for_iv_upper_bound_seed(%ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %ub step %c1 {
    // CHECK: affine_seed_dependency = "seed0 = ?"
    "iree_unregistered.test_affine_seed_dependency"(%iv, %ub) : (index, index) -> ()
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_iv_seed_dependent_step
func.func @scf_for_iv_seed_dependent_step(%step: index) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  scf.for %iv = %c0 to %c16 step %step {
    // CHECK: affine_seed_dependency = "seed0 = ?"
    "iree_unregistered.test_affine_seed_dependency"(%iv, %step) : (index, index) -> ()
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_iter_arg_tracks_loop_carried_dependency
func.func @scf_for_iter_arg_tracks_loop_carried_dependency(%arg0: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> (index) {
    // CHECK: affine_seed_dependency = "seed0 = 1"
    "iree_unregistered.test_affine_seed_dependency"(%iter, %arg0) : (index, index) -> ()
    scf.yield %iter : index
  }
  return %result : index
}

// -----

// CHECK-LABEL: @scf_forall_iv_and_capture
func.func @scf_forall_iv_and_capture(%arg0: index) {
  scf.forall (%iv) in (4) {
    // The forall IV is independent of the map-store-style seed, while an
    // affine expression that also uses the seed still tracks the seed.
    %0 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg0, %iv)
    // CHECK: affine_seed_dependency = "seed0 = 0"
    "iree_unregistered.test_affine_seed_dependency"(%iv, %arg0) : (index, index) -> ()
    // CHECK: affine_seed_dependency = "seed0 = 1"
    "iree_unregistered.test_affine_seed_dependency"(%0, %arg0) : (index, index) -> ()
  }
  return
}

// -----

// CHECK-LABEL: @scf_parallel_multi_iv
func.func @scf_parallel_multi_iv(%arg0: index, %arg1: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ub0 = affine.apply affine_map<(d0) -> (d0 + 4)>(%arg0)
  scf.parallel (%iv0, %iv1) = (%arg0, %c0) to (%ub0, %arg1) step (%c1, %c1) {
    // The first IV has seed-dependent lb/ub with seed-independent extent.
    // CHECK: affine_seed_dependency = "seed0 = 1, seed1 = 0"
    "iree_unregistered.test_affine_seed_dependency"(%iv0, %arg0, %arg1) : (index, index, index) -> ()
    // The second IV has a seed-dependent extent, so it is conservative.
    // CHECK: affine_seed_dependency = "seed0 = 0, seed1 = ?"
    "iree_unregistered.test_affine_seed_dependency"(%iv1, %arg0, %arg1) : (index, index, index) -> ()
  }
  return
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
