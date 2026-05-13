// RUN: iree-opt --iree-codegen-config-tracking-canonicalize --split-input-file %s | FileCheck %s
// RUN: iree-opt --iree-codegen-config-tracking-canonicalize='test-convergence=true' --split-input-file %s | FileCheck %s

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
// CHECK-LABEL: func.func @config_tracking_linalg_operand_drop(
// CHECK-NOT:   tensor.cast
// CHECK:       %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:  ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>)
// CHECK-SAME:  outs(%{{.*}} : tensor<4xf32>)
// CHECK-SAME:  lowering_config
// CHECK-NOT:   tensor.cast
// CHECK:       return %[[GENERIC]]
