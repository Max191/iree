// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 --iree-convert-to-rocdl %s | FileCheck %s

// Test permlane lowering on gfx950.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
module {
  func.func @test_permlane_16_32_lowering() {
    %c0  = arith.constant 0 : index
    %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<1xi32>

    %tid = gpu.thread_id x
    %val = arith.index_castui %tid : index to i32

    // Emits rocdl.permlane*.swap on gfx950.
    %p32 = amdgpu.permlane_swap %val 32 : i32
    %a32 = arith.addi %val, %p32 : i32
    %p16 = amdgpu.permlane_swap %a32 16 : i32
    %sum = arith.addi %a32, %p16 : i32

    %is0 = arith.cmpi eq, %tid, %c0 : index
    scf.if %is0 {
      memref.store %sum, %out[%c0] : memref<1xi32>
    }
    return
  }
}

// CHECK-LABEL: llvm.func @test_permlane_16_32_lowering
// CHECK: rocdl.permlane32.swap
// CHECK: rocdl.permlane16.swap

// -----

module {
  func.func @global_subgroup_barrier() {
    iree_gpu.global_subgroup_barrier
    return
  }
}

// CHECK-LABEL: llvm.func @global_subgroup_barrier
//       CHECK:   rocdl.s.barrier

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
module {
  // Lower gpu.num_subgroups to ceildiv(thread_count, subgroup_size) through
  // the existing block_dim and subgroup_size ROCDL lowerings.
  func.func @num_subgroups_lowering() {
    %c0 = arith.constant 0 : index
    %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<1xi32>
    %num_subgroups = gpu.num_subgroups : index
    %num_subgroups_i32 = arith.index_castui %num_subgroups : index to i32
    memref.store %num_subgroups_i32, %out[%c0] : memref<1xi32>
    return
  }
}

// CHECK-LABEL: llvm.func @num_subgroups_lowering
//   CHECK-NOT: gpu.num_subgroups
//       CHECK: llvm.call @__ockl_get_local_size(
//       CHECK: llvm.call @__ockl_get_local_size(
//       CHECK: llvm.call @__ockl_get_local_size(
//       CHECK: rocdl.wavefrontsize
//       CHECK: arith.ceildivui

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
module {
  // Known function block sizes should fold block_dim operands before ROCDL
  // lowering instead of emitting runtime local-size queries.
  func.func @num_subgroups_known_block_size()
      attributes {gpu.known_block_size = array<i32: 8, 4, 2>} {
    %c0 = arith.constant 0 : index
    %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<1xi32>
    %num_subgroups = gpu.num_subgroups : index
    %num_subgroups_i32 = arith.index_castui %num_subgroups : index to i32
    memref.store %num_subgroups_i32, %out[%c0] : memref<1xi32>
    return
  }
}

// CHECK-LABEL: llvm.func @num_subgroups_known_block_size
//   CHECK-NOT: __ockl_get_local_size
//       CHECK: rocdl.wavefrontsize
//       CHECK: arith.ceildivui

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
module {
  // The upper_bound on gpu.num_subgroups bounds the result quotient. The
  // expanded arithmetic intentionally has no equivalent attribute to forward.
  func.func @num_subgroups_upper_bound() {
    %c0 = arith.constant 0 : index
    %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<1xi32>
    %num_subgroups = gpu.num_subgroups upper_bound 8 : index
    %num_subgroups_i32 = arith.index_castui %num_subgroups : index to i32
    memref.store %num_subgroups_i32, %out[%c0] : memref<1xi32>
    return
  }
}

// CHECK-LABEL: llvm.func @num_subgroups_upper_bound
//   CHECK-NOT: gpu.num_subgroups
//       CHECK: rocdl.wavefrontsize
//       CHECK: arith.ceildivui

// -----

// Verify that arith.truncf f32 to bf16 is NOT expanded on gfx950, which has
// native bf16 conversion instructions (v_cvt_pk_bf16_f32).
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
module {
  func.func @bf16_truncf_native() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<64xf32>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<64xbf16>
    %val = memref.load %0[%c0] : memref<64xf32>
    %trunc = arith.truncf %val : f32 to bf16
    memref.store %trunc, %1[%c0] : memref<64xbf16>
    return
  }
}
// CHECK-LABEL: llvm.func @bf16_truncf_native
//       CHECK:   llvm.fptrunc
//   CHECK-NOT:   llvm.lshr
//       CHECK:   llvm.return
