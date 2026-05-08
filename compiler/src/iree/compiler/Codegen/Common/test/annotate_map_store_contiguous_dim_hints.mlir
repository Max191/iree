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
// ANNOTATE-DAG: #[[$IDENTITY_ANNOTATED_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// ANNOTATE-LABEL: @identity_hints
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$IDENTITY_ANNOTATED_MAP]]
// VECTORIZE-DAG: #[[$IDENTITY_BASE_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
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
// ANNOTATE-DAG: #[[$UNIT_OFFSET_ANNOTATED_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// ANNOTATE-LABEL: @unit_offset_hints
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$UNIT_OFFSET_ANNOTATED_MAP]]
// VECTORIZE-DAG: #[[$UNIT_OFFSET_BASE_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// VECTORIZE-LABEL: @unit_offset_hints
// VECTORIZE: %[[C2:.+]] = arith.constant 2 : index
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: into %{{.*}}[%{{.*}}, %[[C2]]]
// VECTORIZE-SAME: indexing_maps = [#[[$UNIT_OFFSET_BASE_MAP]]]
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
// ANNOTATE-DAG: #[[$PERMUTED_ANNOTATED_MAP:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// ANNOTATE-LABEL: @permuted_hints
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$PERMUTED_ANNOTATED_MAP]]
// VECTORIZE-DAG: #[[$PERMUTED_BASE_MAP:.+]] = affine_map<(d0, d1) -> (d1, d0)>
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
// ANNOTATE-DAG: #[[$STRIDE_ANNOTATED_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1 * 2)>
// ANNOTATE-LABEL: @stride_keeps_non_contiguous_dim_scattered
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$STRIDE_ANNOTATED_MAP]]
// VECTORIZE-DAG: #[[$STRIDE_BASE_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1 * 2)>
// VECTORIZE-LABEL: @stride_keeps_non_contiguous_dim_scattered
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: indexing_maps = [#[[$STRIDE_BASE_MAP]]]
// VECTORIZE-SAME: vector<2x3xf32>, tensor<2x6xf32> -> tensor<2x6xf32>

// -----

func.func @floordiv_expression_uses_symbol_metadata(
    %input: tensor<4xf32>, %output: tensor<2xf32>
) -> tensor<2xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%idx0)
      iree_linalg_ext.yield %1, %mask : index, i1
  } : tensor<4xf32> into tensor<2xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// ANNOTATE-DAG: #[[$FLOORDIV_ANNOTATED_MAP:.+]] = affine_map<(d0)[s0] -> (s0)>
// ANNOTATE-LABEL: @floordiv_expression_uses_symbol_metadata
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$FLOORDIV_ANNOTATED_MAP]]
// VECTORIZE-DAG: #[[$FLOORDIV_BASE_MAP:.+]] = affine_map<(d0)[s0] -> (s0)>
// VECTORIZE-LABEL: @floordiv_expression_uses_symbol_metadata
// VECTORIZE: iree_linalg_ext.map_store
// VECTORIZE-SAME: transfer_scatter_indexing_map = #[[$FLOORDIV_BASE_MAP]]
// VECTORIZE-NOT: iree_vector_ext.transfer_scatter

// -----

func.func @floordiv_with_offset_uses_symbol_metadata(
    %input: tensor<4xf32>, %output: tensor<3xf32>
) -> tensor<3xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      %1 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%idx0)
      iree_linalg_ext.yield %1, %mask : index, i1
  } : tensor<4xf32> into tensor<3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
// ANNOTATE-DAG: #[[$FLOORDIV_OFFSET_ANNOTATED_MAP:.+]] = affine_map<(d0)[s0] -> (s0)>
// ANNOTATE-LABEL: @floordiv_with_offset_uses_symbol_metadata
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$FLOORDIV_OFFSET_ANNOTATED_MAP]]
// VECTORIZE-DAG: #[[$FLOORDIV_OFFSET_BASE_MAP:.+]] = affine_map<(d0)[s0] -> (s0)>
// VECTORIZE-LABEL: @floordiv_with_offset_uses_symbol_metadata
// VECTORIZE: iree_linalg_ext.map_store
// VECTORIZE-SAME: transfer_scatter_indexing_map = #[[$FLOORDIV_OFFSET_BASE_MAP]]
// VECTORIZE-NOT: iree_vector_ext.transfer_scatter

// -----

// Input dim 1 contributes to both output dims. The transfer_scatter indexing
// map can represent this directly.
func.func @duplicate_candidate_maps_to_both_outputs(
    %input: tensor<2x3xf32>, %output: tensor<3x3xf32>
) -> tensor<3x3xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx1, %idx1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<3x3xf32> -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
// ANNOTATE-DAG: #[[$DUPLICATE_OUTPUT_ANNOTATED_MAP:.+]] = affine_map<(d0, d1) -> (d1, d1)>
// ANNOTATE-LABEL: @duplicate_candidate_maps_to_both_outputs
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$DUPLICATE_OUTPUT_ANNOTATED_MAP]]
// VECTORIZE-DAG: #[[$DUPLICATE_OUTPUT_BASE_MAP:.+]] = affine_map<(d0, d1) -> (d1, d1)>
// VECTORIZE-LABEL: @duplicate_candidate_maps_to_both_outputs
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: indexing_maps = [#[[$DUPLICATE_OUTPUT_BASE_MAP]]]

// -----

// The analysis reasons about final seed coefficients, so duplicate uses that
// simplify to a single unit dependence are not ambiguous.
func.func @duplicate_apply_operand_is_hinted(
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
// ANNOTATE-DAG: #[[$DUPLICATE_ANNOTATED_MAP:.+]] = affine_map<(d0) -> (d0)>
// ANNOTATE-LABEL: @duplicate_apply_operand_is_hinted
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$DUPLICATE_ANNOTATED_MAP]]
// VECTORIZE-DAG: #[[$DUPLICATE_BASE_MAP:.+]] = affine_map<(d0) -> (d0)>
// VECTORIZE-LABEL: @duplicate_apply_operand_is_hinted
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: indexing_maps = [#[[$DUPLICATE_BASE_MAP]]
// VECTORIZE-SAME: vector<3xf32>, tensor<3xf32> -> tensor<3xf32>

// -----

func.func @symbol_operand_is_hinted(
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
// ANNOTATE-DAG: #[[$SYMBOL_ANNOTATED_MAP:.+]] = affine_map<(d0) -> (d0)>
// ANNOTATE-LABEL: @symbol_operand_is_hinted
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$SYMBOL_ANNOTATED_MAP]]
// VECTORIZE-DAG: #[[$SYMBOL_BASE_MAP:.+]] = affine_map<(d0) -> (d0)>
// VECTORIZE-LABEL: @symbol_operand_is_hinted
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: indexing_maps = [#[[$SYMBOL_BASE_MAP]]
// VECTORIZE-SAME: vector<3xf32>, tensor<3xf32> -> tensor<3xf32>

// -----

func.func @chained_apply_is_hinted(
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
// ANNOTATE-DAG: #[[$CHAINED_ANNOTATED_MAP:.+]] = affine_map<(d0) -> (d0)>
// ANNOTATE-LABEL: @chained_apply_is_hinted
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$CHAINED_ANNOTATED_MAP]]
// VECTORIZE-DAG: #[[$CHAINED_BASE_MAP:.+]] = affine_map<(d0) -> (d0)>
// VECTORIZE-LABEL: @chained_apply_is_hinted
// VECTORIZE: %[[C2:.+]] = arith.constant 2 : index
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: into %{{.*}}[%[[C2]]]
// VECTORIZE-SAME: indexing_maps = [#[[$CHAINED_BASE_MAP]]
// VECTORIZE-SAME: vector<3xf32>, tensor<5xf32> -> tensor<5xf32>

// -----

func.func @unknown_seed_dependency_uses_symbol_metadata(
    %input: tensor<2x3xf32>, %output: tensor<6xf32>
) -> tensor<6xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = arith.muli %idx0, %idx1 : index
      iree_linalg_ext.yield %1, %mask : index, i1
  } : tensor<2x3xf32> into tensor<6xf32> -> tensor<6xf32>
  return %0 : tensor<6xf32>
}
// ANNOTATE-DAG: #[[$UNKNOWN_ANNOTATED_MAP:.+]] = affine_map<(d0, d1)[s0] -> (s0)>
// ANNOTATE-LABEL: @unknown_seed_dependency_uses_symbol_metadata
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$UNKNOWN_ANNOTATED_MAP]]
// VECTORIZE-DAG: #[[$UNKNOWN_BASE_MAP:.+]] = affine_map<(d0, d1)[s0] -> (s0)>
// VECTORIZE-LABEL: @unknown_seed_dependency_uses_symbol_metadata
// VECTORIZE: iree_linalg_ext.map_store
// VECTORIZE-SAME: transfer_scatter_indexing_map = #[[$UNKNOWN_BASE_MAP]]
// VECTORIZE-NOT: iree_vector_ext.transfer_scatter

// -----

func.func @multiple_map_stores_are_annotated_independently(
    %input0: tensor<3xf32>, %input1: tensor<2x3xf32>,
    %output0: tensor<3xf32>, %output1: tensor<6xf32>
) -> (tensor<3xf32>, tensor<6xf32>) {
  %0 = iree_linalg_ext.map_store %input0 into %output0 {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %mask : index, i1
  } : tensor<3xf32> into tensor<3xf32> -> tensor<3xf32>
  %1 = iree_linalg_ext.map_store %input1 into %output1 {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %2 = arith.muli %idx0, %idx1 : index
      iree_linalg_ext.yield %2, %mask : index, i1
  } : tensor<2x3xf32> into tensor<6xf32> -> tensor<6xf32>
  return %0, %1 : tensor<3xf32>, tensor<6xf32>
}
// ANNOTATE-DAG: #[[$MULTI_FIRST_ANNOTATED_MAP:.+]] = affine_map<(d0) -> (d0)>
// ANNOTATE-DAG: #[[$MULTI_SECOND_ANNOTATED_MAP:.+]] = affine_map<(d0, d1)[s0] -> (s0)>
// ANNOTATE-LABEL: @multiple_map_stores_are_annotated_independently
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$MULTI_FIRST_ANNOTATED_MAP]]
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$MULTI_SECOND_ANNOTATED_MAP]]
// VECTORIZE-DAG: #[[$MULTI_FIRST_BASE_MAP:.+]] = affine_map<(d0) -> (d0)>
// VECTORIZE-DAG: #[[$MULTI_SECOND_BASE_MAP:.+]] = affine_map<(d0, d1)[s0] -> (s0)>
// VECTORIZE-LABEL: @multiple_map_stores_are_annotated_independently
// VECTORIZE: iree_vector_ext.transfer_scatter
// VECTORIZE-SAME: indexing_maps = [#[[$MULTI_FIRST_BASE_MAP]]
// VECTORIZE: iree_linalg_ext.map_store
// VECTORIZE-SAME: transfer_scatter_indexing_map = #[[$MULTI_SECOND_BASE_MAP]]
// VECTORIZE-NOT: iree_vector_ext.transfer_scatter

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
