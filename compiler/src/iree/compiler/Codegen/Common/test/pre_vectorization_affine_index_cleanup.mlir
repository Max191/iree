// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-pre-vectorization-affine-index-cleanup))" --split-input-file %s | FileCheck %s --check-prefix=CLEANUP
// Also verify that the convergence-test mode succeeds on the same cases.
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-pre-vectorization-affine-index-cleanup{test-convergence=true}))" --split-input-file %s | FileCheck %s --check-prefix=CLEANUP

func.func @motivating_map_store_chain(
    %input: tensor<4x8xf32>, %output: tensor<4x10xf32>
) -> tensor<4x10xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %linear = affine.linearize_index disjoint [%idx0, %idx1] by (4, 8) : index
      %delinearized:2 = affine.delinearize_index %linear into (4, 8) : index, index
      %offset = affine.apply affine_map<(d0) -> (d0 + 2)>(%delinearized#1)
      iree_linalg_ext.yield %delinearized#0, %offset, %mask
          : index, index, i1
  } : tensor<4x8xf32> into tensor<4x10xf32> -> tensor<4x10xf32>
  return %0 : tensor<4x10xf32>
}
// CLEANUP-LABEL: func.func @motivating_map_store_chain
// CLEANUP:       iree_linalg_ext.map_store
// CLEANUP-NOT:   affine.linearize_index
// CLEANUP-NOT:   affine.delinearize_index
// CLEANUP:       affine.apply
// CLEANUP:       iree_linalg_ext.yield %{{.*}}, %{{.*}}, %{{.*}} : index, index, i1

// -----

func.func @config_tracking_linalg_operand_drop(
    %lhs: tensor<4xf32>, %rhs: tensor<4xf32>, %out: tensor<4xf32>
) -> tensor<4xf32> {
  %lhs_dyn = tensor.cast %lhs : tensor<4xf32> to tensor<?xf32>
  %rhs_dyn = tensor.cast %rhs : tensor<4xf32> to tensor<?xf32>
  %out_dyn = tensor.cast %out : tensor<4xf32> to tensor<?xf32>
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"],
      lowering_config = #iree_codegen.lowering_config<tile_sizes = []>
    } ins(%lhs_dyn, %rhs_dyn : tensor<?xf32>, tensor<?xf32>)
      outs(%out_dyn : tensor<?xf32>) {
    ^bb0(%lhs_elem: f32, %rhs_elem: f32, %out_elem: f32):
      %sum = arith.addf %lhs_elem, %out_elem : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
  %1 = tensor.cast %0 : tensor<?xf32> to tensor<4xf32>
  return %1 : tensor<4xf32>
}
// CLEANUP-LABEL: func.func @config_tracking_linalg_operand_drop(
// CLEANUP-NOT:   tensor.cast
// CLEANUP:       %[[GENERIC:.+]] = linalg.generic
// CLEANUP-SAME:  ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>)
// CLEANUP-SAME:  outs(%{{.*}} : tensor<4xf32>)
// CLEANUP-SAME:  lowering_config
// CLEANUP-NOT:   tensor.cast
// CLEANUP:       return %[[GENERIC]]

// -----

func.func @exact_inverse_with_outer_bound(
    %a: index, %b: index, %c: index
) -> (index, index, index) {
  %0 = affine.linearize_index disjoint [%a, %b, %c] by (2, 3, 4) : index
  %1:3 = affine.delinearize_index %0 into (2, 3, 4) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}
// CLEANUP-LABEL: func.func @exact_inverse_with_outer_bound(
// CLEANUP-SAME:      %[[A:[A-Za-z0-9_]+]]: index, %[[B:[A-Za-z0-9_]+]]: index, %[[C:[A-Za-z0-9_]+]]: index)
// CLEANUP-NOT:   affine.linearize_index
// CLEANUP-NOT:   affine.delinearize_index
// CLEANUP:       return %[[A]], %[[B]], %[[C]]

// -----

func.func @exact_inverse_without_outer_bound(
    %a: index, %b: index, %c: index
) -> (index, index, index) {
  %0 = affine.linearize_index disjoint [%a, %b, %c] by (3, 4) : index
  %1:3 = affine.delinearize_index %0 into (3, 4) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}
// CLEANUP-LABEL: func.func @exact_inverse_without_outer_bound(
// CLEANUP-SAME:      %[[A:[A-Za-z0-9_]+]]: index, %[[B:[A-Za-z0-9_]+]]: index, %[[C:[A-Za-z0-9_]+]]: index)
// CLEANUP-NOT:   affine.linearize_index
// CLEANUP-NOT:   affine.delinearize_index
// CLEANUP:       return %[[A]], %[[B]], %[[C]]

// -----

func.func @many_to_one_tail(
    %a: index, %b: index, %c: index
) -> (index, index) {
  %0 = affine.linearize_index disjoint [%a, %b, %c] by (4, 8, 8) : index
  %1:2 = affine.delinearize_index %0 into (4, 64) : index, index
  return %1#0, %1#1 : index, index
}
// CLEANUP-LABEL: func.func @many_to_one_tail(
// CLEANUP-SAME:      %[[A:[A-Za-z0-9_]+]]: index, %[[B:[A-Za-z0-9_]+]]: index, %[[C:[A-Za-z0-9_]+]]: index)
// CLEANUP:       %[[LIN:.+]] = affine.linearize_index disjoint [%[[B]], %[[C]]] by (8, 8)
// CLEANUP:       return %[[A]], %[[LIN]]

// -----

func.func @one_to_many_tail(
    %a: index, %b: index
) -> (index, index, index) {
  %0 = affine.linearize_index disjoint [%a, %b] by (4, 64) : index
  %1:3 = affine.delinearize_index %0 into (4, 8, 8) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}
// CLEANUP-LABEL: func.func @one_to_many_tail(
// CLEANUP-SAME:      %[[A:[A-Za-z0-9_]+]]: index, %[[B:[A-Za-z0-9_]+]]: index)
// CLEANUP:       %[[DELIN:.+]]:2 = affine.delinearize_index %[[B]] into (8, 8)
// CLEANUP:       return %[[A]], %[[DELIN]]#0, %[[DELIN]]#1

// -----

func.func @partial_tail(
    %a: index, %b: index, %c: index, %d: index
) -> (index, index, index) {
  %0 = affine.linearize_index disjoint [%a, %b, %c, %d] by (5, 3, 4, 8) : index
  %1:3 = affine.delinearize_index %0 into (7, 9, 32) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}
// CLEANUP-LABEL: func.func @partial_tail(
// CLEANUP-SAME:      %[[A:[A-Za-z0-9_]+]]: index, %[[B:[A-Za-z0-9_]+]]: index, %[[C:[A-Za-z0-9_]+]]: index, %[[D:[A-Za-z0-9_]+]]: index)
// CLEANUP:       %[[RESLIN:.+]] = affine.linearize_index disjoint [%[[A]], %[[B]]] by (5, 3)
// CLEANUP:       %[[RESDELIN:.+]]:2 = affine.delinearize_index %[[RESLIN]] into (7, 9)
// CLEANUP:       %[[TAILLIN:.+]] = affine.linearize_index disjoint [%[[C]], %[[D]]] by (4, 8)
// CLEANUP:       return %[[RESDELIN]]#0, %[[RESDELIN]]#1, %[[TAILLIN]]

// -----

func.func @no_outer_bound_many_to_one_tail(
    %a: index, %b: index, %c: index
) -> (index, index) {
  %0 = affine.linearize_index disjoint [%a, %b, %c] by (8, 8) : index
  %1:2 = affine.delinearize_index %0 into (64) : index, index
  return %1#0, %1#1 : index, index
}
// CLEANUP-LABEL: func.func @no_outer_bound_many_to_one_tail(
// CLEANUP-SAME:      %[[A:[A-Za-z0-9_]+]]: index, %[[B:[A-Za-z0-9_]+]]: index, %[[C:[A-Za-z0-9_]+]]: index)
// CLEANUP:       %[[LIN:.+]] = affine.linearize_index disjoint [%[[B]], %[[C]]] by (8, 8)
// CLEANUP:       return %[[A]], %[[LIN]]

// -----

func.func @unit_and_trivial_cleanup(
    %arg0: index, %arg1: index
) -> (index, index, index) {
  %0:3 = affine.delinearize_index %arg0 into (1, %arg1, 1) : index, index, index
  %1 = affine.linearize_index [%arg1] by (42) : index
  return %0#0, %0#2, %1 : index, index, index
}
// CLEANUP-LABEL: func.func @unit_and_trivial_cleanup(
// CLEANUP-SAME:      %{{.*}}: index, %[[ARG1:[A-Za-z0-9_]+]]: index)
// CLEANUP-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CLEANUP-NOT:   affine.linearize_index
// CLEANUP-NOT:   affine.delinearize_index
// CLEANUP:       return %[[C0]], %[[C0]], %[[ARG1]]

// -----

func.func @non_disjoint_negative(
    %a: index, %b: index, %c: index
) -> (index, index) {
  %0 = affine.linearize_index [%a, %b, %c] by (4, 8, 8) : index
  %1:2 = affine.delinearize_index %0 into (4, 64) : index, index
  return %1#0, %1#1 : index, index
}
// CLEANUP-LABEL: func.func @non_disjoint_negative(
// CLEANUP:       affine.linearize_index [
// CLEANUP:       affine.delinearize_index

// -----

func.func @product_mismatch_negative(
    %a: index, %b: index, %c: index
) -> (index, index) {
  %0 = affine.linearize_index disjoint [%a, %b, %c] by (4, 8, 8) : index
  %1:2 = affine.delinearize_index %0 into (4, 63) : index, index
  return %1#0, %1#1 : index, index
}
// CLEANUP-LABEL: func.func @product_mismatch_negative(
// CLEANUP:       affine.linearize_index disjoint
// CLEANUP:       affine.delinearize_index
