// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-annotate-map-store-contiguous-dim-hints))" --split-input-file %s | FileCheck %s --check-prefix=ANNOTATE
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-annotate-map-store-contiguous-dim-hints,iree-codegen-generic-vectorization{enable-vector-masking=false vectorize-map-store=true}))" --split-input-file %s | FileCheck %s --check-prefix=VECTORIZE

func.func @identity_hints(
    %input: tensor<2x3xf32>, %output: tensor<2x3xf32>
) -> tensor<2x3xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
// ANNOTATE-LABEL: @identity_hints
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: contiguous_dim_hints = array<i64: 0, 0, 1, 1>
// VECTORIZE-DAG: #[[$IDENTITY_BASE_MAP:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 + s0, d1 + s1)>
// VECTORIZE-LABEL: @identity_hints
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: indexing_maps = [#[[$IDENTITY_BASE_MAP]]

// -----

func.func @unit_offset_hints(
    %input: tensor<2x3xf32>, %output: tensor<2x5xf32>
) -> tensor<2x5xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 + 2)>(%idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x5xf32> -> tensor<2x5xf32>
  return %0 : tensor<2x5xf32>
}
// ANNOTATE-LABEL: @unit_offset_hints
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: contiguous_dim_hints = array<i64: 0, 0, 1, 1>
// VECTORIZE-LABEL: @unit_offset_hints
// VECTORIZE: arith.constant dense<2> : vector<2xindex>
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: tensor<2x5xf32> -> tensor<2x5xf32>

// -----

func.func @permuted_hints(
    %input: tensor<2x3xf32>, %output: tensor<3x2xf32>
) -> tensor<3x2xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx1, %idx0, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<3x2xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
// ANNOTATE-LABEL: @permuted_hints
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: contiguous_dim_hints = array<i64: 0, 1, 1, 0>
// VECTORIZE-DAG: #[[$PERMUTED_BASE_MAP:.+]] = affine_map<(d0, d1)[s0, s1] -> (d1 + s0, d0 + s1)>
// VECTORIZE-LABEL: @permuted_hints
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: indexing_maps = [#[[$PERMUTED_BASE_MAP]]

// -----

func.func @stride_keeps_non_contiguous_dim_scattered(
    %input: tensor<2x3xf32>, %output: tensor<2x6xf32>
) -> tensor<2x6xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%idx1)
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x6xf32> -> tensor<2x6xf32>
  return %0 : tensor<2x6xf32>
}
// ANNOTATE-LABEL: @stride_keeps_non_contiguous_dim_scattered
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: contiguous_dim_hints = array<i64: 0, 0>
// VECTORIZE-LABEL: @stride_keeps_non_contiguous_dim_scattered
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: vector<3xindex>, vector<3xindex>

// -----

// Input dim 1 is a unit candidate for both output dims, so it is not
// unambiguous enough to represent as a contiguous transfer_scatter dimension.
func.func @duplicate_candidate_is_not_hinted(
    %input: tensor<2x3xf32>, %output: tensor<3x3xf32>
) -> tensor<3x3xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx1, %idx1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<3x3xf32> -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
// ANNOTATE-LABEL: @duplicate_candidate_is_not_hinted
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-NOT: contiguous_dim_hints
// VECTORIZE-LABEL: @duplicate_candidate_is_not_hinted
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: vector<3xindex>

// -----

// Even if the affine map ignores one operand, using the same input block arg
// twice in the affine.apply operands is intentionally rejected as ambiguous.
func.func @duplicate_apply_operand_is_not_hinted(
    %input: tensor<3xf32>, %output: tensor<3xf32>
) -> tensor<3xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0, d1) -> (d0)>(%idx0, %idx0)
      iree_linalg_ext.yield %1, %mask : index, i1
  } : tensor<3xf32> into tensor<3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
// ANNOTATE-LABEL: @duplicate_apply_operand_is_not_hinted
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-NOT: contiguous_dim_hints
// VECTORIZE-LABEL: @duplicate_apply_operand_is_not_hinted
// VECTORIZE: iree_linalg_ext.map_store
// VECTORIZE-NOT: transfer_scatter

// -----

func.func @symbol_operand_is_not_hinted(
    %input: tensor<3xf32>, %output: tensor<3xf32>
) -> tensor<3xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<()[s0] -> (s0)>()[%idx0]
      iree_linalg_ext.yield %1, %mask : index, i1
  } : tensor<3xf32> into tensor<3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
// ANNOTATE-LABEL: @symbol_operand_is_not_hinted
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-NOT: contiguous_dim_hints
// VECTORIZE-LABEL: @symbol_operand_is_not_hinted
// VECTORIZE: iree_linalg_ext.map_store
// VECTORIZE-NOT: transfer_scatter

// -----

func.func @chained_apply_is_not_hinted(
    %input: tensor<3xf32>, %output: tensor<5xf32>
) -> tensor<5xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%idx0)
      %2 = affine.apply affine_map<(d0) -> (d0 + 1)>(%1)
      iree_linalg_ext.yield %2, %mask : index, i1
  } : tensor<3xf32> into tensor<5xf32> -> tensor<5xf32>
  return %0 : tensor<5xf32>
}
// ANNOTATE-LABEL: @chained_apply_is_not_hinted
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-NOT: contiguous_dim_hints
// VECTORIZE-LABEL: @chained_apply_is_not_hinted
// VECTORIZE: iree_linalg_ext.map_store
// VECTORIZE-NOT: transfer_scatter

// -----

func.func @existing_hints_are_preserved(
    %input: tensor<2x3xf32>, %output: tensor<2x3xf32>
) -> tensor<2x3xf32> {
  %0 = iree_linalg_ext.map_store {contiguous_dim_hints = array<i64: 1, 1>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
// ANNOTATE-LABEL: @existing_hints_are_preserved
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: contiguous_dim_hints = array<i64: 1, 1>
// VECTORIZE-LABEL: @existing_hints_are_preserved
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: vector<2xindex>
