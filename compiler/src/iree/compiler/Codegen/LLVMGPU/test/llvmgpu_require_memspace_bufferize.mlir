// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-llvmgpu-test-require-memspace-bufferization-pipeline))" --split-input-file --verify-diagnostics %s | FileCheck %s

// These alloc_tensor cases intentionally mirror llvmgpu_bufferize.mlir to pin
// the required-memory-space allocation hook to the same PCF scope behavior.
func.func @require_memspace_alloc_tensor_in_lane_pcf() {
  pcf.generic scope(#iree_gpu.lane_scope)
    execute[%id: index, %n: index] {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.0> : vector<4xf32>
    %alloc = bufferization.alloc_tensor() : tensor<4xf32>
    %written = vector.transfer_write %cst, %alloc[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>
    %element = tensor.extract %written[%c0] : tensor<4xf32>
    util.optimization_barrier %element : f32
    pcf.return
  }
  return
}

// CHECK-LABEL: func.func @require_memspace_alloc_tensor_in_lane_pcf
//       CHECK:   pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK:     %[[ALLOC:.+]] = memref.alloca() : memref<4xf32, #gpu.address_space<private>>
//       CHECK:     vector.transfer_write %{{.*}}, %[[ALLOC]]

// -----

func.func @require_memspace_alloc_tensor_in_subgroup_pcf() {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    execute[%id: index, %n: index] {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.0> : vector<4xf32>
    %alloc = bufferization.alloc_tensor() : tensor<4xf32>
    %written = vector.transfer_write %cst, %alloc[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>
    %element = tensor.extract %written[%c0] : tensor<4xf32>
    util.optimization_barrier %element : f32
    pcf.return
  }
  return
}

// CHECK-LABEL: func.func @require_memspace_alloc_tensor_in_subgroup_pcf
//       CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:     %[[ALLOC:.+]] = memref.alloc() : memref<4xf32, #gpu.address_space<workgroup>>
//       CHECK:     vector.transfer_write %{{.*}}, %[[ALLOC]]

// -----

func.func @require_memspace_alloc_tensor_in_lane_pcf_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pcf.loop scope(#iree_gpu.lane_scope) count(%c1)
    execute[%id: index] {
    %cst = arith.constant dense<0.0> : vector<4xf32>
    %alloc = bufferization.alloc_tensor() : tensor<4xf32>
    %written = vector.transfer_write %cst, %alloc[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>
    %element = tensor.extract %written[%c0] : tensor<4xf32>
    util.optimization_barrier %id, %element : index, f32
    pcf.return
  }
  return
}

// CHECK-LABEL: func.func @require_memspace_alloc_tensor_in_lane_pcf_loop
//       CHECK:   pcf.loop scope(#iree_gpu.lane_scope)
//       CHECK:     %[[ALLOC:.+]] = memref.alloca() : memref<4xf32, #gpu.address_space<private>>
//       CHECK:     vector.transfer_write %{{.*}}, %[[ALLOC]]

// -----

func.func @require_memspace_alloc_tensor_in_subgroup_pcf_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pcf.loop scope(#iree_gpu.subgroup_scope) count(%c1)
    execute[%id: index] {
    %cst = arith.constant dense<0.0> : vector<4xf32>
    %alloc = bufferization.alloc_tensor() : tensor<4xf32>
    %written = vector.transfer_write %cst, %alloc[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>
    %element = tensor.extract %written[%c0] : tensor<4xf32>
    util.optimization_barrier %id, %element : index, f32
    pcf.return
  }
  return
}

// CHECK-LABEL: func.func @require_memspace_alloc_tensor_in_subgroup_pcf_loop
//       CHECK:   pcf.loop scope(#iree_gpu.subgroup_scope)
//       CHECK:     %[[ALLOC:.+]] = memref.alloc() : memref<4xf32, #gpu.address_space<workgroup>>
//       CHECK:     vector.transfer_write %{{.*}}, %[[ALLOC]]

// -----

func.func @require_memspace_inner_lane_over_outer_subgroup_pcf() {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    execute[%subgroup_id: index, %num_subgroups: index] {
    pcf.generic scope(#iree_gpu.lane_scope)
      execute[%lane_id: index, %subgroup_size: index] {
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.0> : vector<4xf32>
      %alloc = bufferization.alloc_tensor() : tensor<4xf32>
      %written = vector.transfer_write %cst, %alloc[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>
      %element = tensor.extract %written[%c0] : tensor<4xf32>
      util.optimization_barrier %element : f32
      pcf.return
    }
    pcf.return
  }
  return
}

// CHECK-LABEL: func.func @require_memspace_inner_lane_over_outer_subgroup_pcf
//       CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:     pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK:       %[[ALLOC:.+]] = memref.alloca() : memref<4xf32, #gpu.address_space<private>>
//       CHECK:       vector.transfer_write %{{.*}}, %[[ALLOC]]

// -----

func.func @require_memspace_thread_forall_over_outer_subgroup_pcf() {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    execute[%subgroup_id: index, %num_subgroups: index] {
    scf.forall (%thread_id) in (1) {
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.0> : vector<4xf32>
      %alloc = bufferization.alloc_tensor() : tensor<4xf32>
      %written = vector.transfer_write %cst, %alloc[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>
      %element = tensor.extract %written[%c0] : tensor<4xf32>
      util.optimization_barrier %element : f32
    } {mapping = [#gpu.thread<x>]}
    pcf.return
  }
  return
}

// CHECK-LABEL: func.func @require_memspace_thread_forall_over_outer_subgroup_pcf
//       CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:     scf.forall
//       CHECK:       %[[ALLOC:.+]] = memref.alloca() : memref<4xf32, #gpu.address_space<private>>
//       CHECK:       vector.transfer_write %{{.*}}, %[[ALLOC]]

// -----

func.func @require_memspace_alloc_tensor_without_gpu_context_defaults_private() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.0> : vector<4xf32>
  %alloc = bufferization.alloc_tensor() : tensor<4xf32>
  %written = vector.transfer_write %cst, %alloc[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>
  %element = tensor.extract %written[%c0] : tensor<4xf32>
  util.optimization_barrier %element : f32
  return
}

// CHECK-LABEL: func.func @require_memspace_alloc_tensor_without_gpu_context_defaults_private
//       CHECK:   %[[ALLOC:.+]] = memref.alloca() : memref<4xf32, #gpu.address_space<private>>
//       CHECK:   vector.transfer_write %{{.*}}, %[[ALLOC]]

// -----

func.func @require_memspace_descriptor_write_in_lane_pcf(
    %arg0: memref<4xf32, #hal.descriptor_type<storage_buffer>>) {
  pcf.generic scope(#iree_gpu.lane_scope)
    execute[%id: index, %n: index] {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant dense<0.0> : vector<2xf32>
    %tensor = bufferization.to_tensor %arg0 restrict : memref<4xf32, #hal.descriptor_type<storage_buffer>> to tensor<4xf32>
    %written = vector.transfer_write %cst, %tensor[%c0] {in_bounds = [true]} : vector<2xf32>, tensor<4xf32>
    %element = tensor.extract %written[%c3] : tensor<4xf32>
    util.optimization_barrier %element : f32
    pcf.return
  }
  return
}

// CHECK-LABEL: func.func @require_memspace_descriptor_write_in_lane_pcf
//       CHECK:   pcf.generic scope(#iree_gpu.lane_scope)
//       CHECK:     %[[ALLOC:.+]] = memref.alloca() : memref<4xf32, #gpu.address_space<private>>
//       CHECK:     memref.copy %{{.*}}, %[[ALLOC]]
//       CHECK:     vector.transfer_write %{{.*}}, %[[ALLOC]]

// -----

func.func @require_memspace_descriptor_write_in_subgroup_pcf(
    %arg0: memref<4xf32, #hal.descriptor_type<storage_buffer>>) {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    execute[%id: index, %n: index] {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant dense<0.0> : vector<2xf32>
    %tensor = bufferization.to_tensor %arg0 restrict : memref<4xf32, #hal.descriptor_type<storage_buffer>> to tensor<4xf32>
    %written = vector.transfer_write %cst, %tensor[%c0] {in_bounds = [true]} : vector<2xf32>, tensor<4xf32>
    %element = tensor.extract %written[%c3] : tensor<4xf32>
    util.optimization_barrier %element : f32
    pcf.return
  }
  return
}

// CHECK-LABEL: func.func @require_memspace_descriptor_write_in_subgroup_pcf
//       CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
//       CHECK:     %[[ALLOC:.+]] = memref.alloc() : memref<4xf32, #gpu.address_space<workgroup>>
//       CHECK:     memref.copy %{{.*}}, %[[ALLOC]]
//       CHECK:     vector.transfer_write %{{.*}}, %[[ALLOC]]

// -----

func.func @require_memspace_descriptor_without_context_fails(
    %arg0: memref<4xf32, #hal.descriptor_type<storage_buffer>>) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant dense<0.0> : vector<2xf32>
  %tensor = bufferization.to_tensor %arg0 restrict : memref<4xf32, #hal.descriptor_type<storage_buffer>> to tensor<4xf32>
  // expected-error @+1 {{failed to bufferize op}}
  %written = vector.transfer_write %cst, %tensor[%c0] {in_bounds = [true]} : vector<2xf32>, tensor<4xf32>
  %element = tensor.extract %written[%c3] : tensor<4xf32>
  util.optimization_barrier %element : f32
  return
}
