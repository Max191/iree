// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-annotate-map-store-contiguous-dim-hints))" --split-input-file %s | FileCheck %s --check-prefix=ANNOTATE

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
// ANNOTATE-DAG: #[[$IDENTITY_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// ANNOTATE-LABEL: @identity_hints
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$IDENTITY_MAP]]

// -----

func.func @constant_affine_offset_hints(
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
// The metadata intentionally captures only the input-index contribution; the
// constant offset is materialized later from the original yielded index.
// ANNOTATE-DAG: #[[$OFFSET_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// ANNOTATE-LABEL: @constant_affine_offset_hints
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$OFFSET_MAP]]

// -----

func.func @external_offset_hints(
    %input: tensor<3xf32>, %output: tensor<?xf32>, %offset: index
) -> tensor<?xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      %1 = arith.addi %idx0, %offset : index
      iree_linalg_ext.yield %1, %mask : index, i1
  } : tensor<3xf32> into tensor<?xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// The dynamic offset is seed-independent, so it is kept out of the map and
// recovered by the vectorizer from the full output index expression.
// ANNOTATE-DAG: #[[$EXTERNAL_OFFSET_MAP:.+]] = affine_map<(d0) -> (d0)>
// ANNOTATE-LABEL: @external_offset_hints
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$EXTERNAL_OFFSET_MAP]]

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
// ANNOTATE-DAG: #[[$PERMUTED_MAP:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// ANNOTATE-LABEL: @permuted_hints
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$PERMUTED_MAP]]

// -----

func.func @strided_hints(
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
// ANNOTATE-DAG: #[[$STRIDED_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1 * 2)>
// ANNOTATE-LABEL: @strided_hints
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$STRIDED_MAP]]

// -----

func.func @duplicate_input_dim_hints(
    %input: tensor<2x3xf32>, %output: tensor<3x3xf32>
) -> tensor<3x3xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx1, %idx1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<3x3xf32> -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
// ANNOTATE-DAG: #[[$DUPLICATE_MAP:.+]] = affine_map<(d0, d1) -> (d1, d1)>
// ANNOTATE-LABEL: @duplicate_input_dim_hints
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$DUPLICATE_MAP]]

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
// ANNOTATE-DAG: #[[$UNKNOWN_MAP:.+]] = affine_map<(d0, d1)[s0] -> (s0)>
// ANNOTATE-LABEL: @unknown_seed_dependency_uses_symbol_metadata
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$UNKNOWN_MAP]]

// -----

func.func @mixed_affine_and_unknown_output_dims(
    %input: tensor<2x3xf32>, %output: tensor<2x6xf32>
) -> tensor<2x6xf32> {
  %0 = iree_linalg_ext.map_store %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      %1 = arith.muli %idx0, %idx1 : index
      iree_linalg_ext.yield %idx0, %1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x6xf32> -> tensor<2x6xf32>
  return %0 : tensor<2x6xf32>
}
// ANNOTATE-DAG: #[[$MIXED_MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0, s0)>
// ANNOTATE-LABEL: @mixed_affine_and_unknown_output_dims
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$MIXED_MAP]]

// -----

func.func @multiple_map_stores_are_annotated_independently(
    %input0: tensor<2x3xf32>, %input1: tensor<2x3xf32>,
    %output0: tensor<2x3xf32>, %output1: tensor<3x2xf32>
) -> (tensor<2x3xf32>, tensor<3x2xf32>) {
  %0 = iree_linalg_ext.map_store %input0 into %output0 {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x3xf32> -> tensor<2x3xf32>
  %1 = iree_linalg_ext.map_store %input1 into %output1 {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx1, %idx0, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<3x2xf32> -> tensor<3x2xf32>
  return %0, %1 : tensor<2x3xf32>, tensor<3x2xf32>
}
// ANNOTATE-DAG: #[[$MULTI_IDENTITY_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// ANNOTATE-DAG: #[[$MULTI_PERMUTED_MAP:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// ANNOTATE-LABEL: @multiple_map_stores_are_annotated_independently
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$MULTI_IDENTITY_MAP]]
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$MULTI_PERMUTED_MAP]]

// -----

func.func @existing_vectorization_hint_is_preserved(
    %input: tensor<2x3xf32>, %output: tensor<2x3xf32>
) -> tensor<2x3xf32> {
  %0 = iree_linalg_ext.map_store {contiguous_dim_hints = array<i64: 0, 0>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<2x3xf32> -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
// ANNOTATE-LABEL: @existing_vectorization_hint_is_preserved
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: contiguous_dim_hints = array<i64: 0, 0>
// ANNOTATE-NOT: transfer_scatter_indexing_map

// -----

func.func @existing_transfer_scatter_indexing_map_is_preserved(
    %input: tensor<2x3xf32>, %output: tensor<5x2xf32>
) -> tensor<5x2xf32> {
  %0 = iree_linalg_ext.map_store {transfer_scatter_indexing_map = affine_map<(d0, d1) -> (d0 + d1, d0)>} %input into %output {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx1, %idx0, %mask : index, index, i1
  } : tensor<2x3xf32> into tensor<5x2xf32> -> tensor<5x2xf32>
  return %0 : tensor<5x2xf32>
}
// ANNOTATE-DAG: #[[$EXISTING_MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1, d0)>
// ANNOTATE-LABEL: @existing_transfer_scatter_indexing_map_is_preserved
// ANNOTATE: iree_linalg_ext.map_store
// ANNOTATE-SAME: transfer_scatter_indexing_map = #[[$EXISTING_MAP]]
// ANNOTATE-NOT: contiguous_dim_hints
