// Extracted from compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_vector_distribute_gfx942.mlir.
// This is the single full ROCm dispatch containing map_store immediately before VectorDistribute lowering.

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = false>}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @matmul_map_store {
hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @matmul_map_store layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2)
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul_map_store() attributes {translation_info = #translation} {
      %true = arith.constant true
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<256x256xf16, #hal.descriptor_type<storage_buffer>>
      %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<256x256xf16, #hal.descriptor_type<storage_buffer>> to memref<256x256xf16, #amdgpu.address_space<fat_raw_buffer>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<256x256xf16, #hal.descriptor_type<storage_buffer>>
      %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<256x256xf16, #hal.descriptor_type<storage_buffer>> to memref<256x256xf16, #amdgpu.address_space<fat_raw_buffer>>
      %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<2x16x8x4x4x4x4xf32, #hal.descriptor_type<storage_buffer>>
      %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<2x16x8x4x4x4x4xf32, #hal.descriptor_type<storage_buffer>> to memref<2x16x8x4x4x4x4xf32, #amdgpu.address_space<fat_raw_buffer>>
      %6 = iree_codegen.load_from_buffer %1 : memref<256x256xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<256x256xf16>
      %7 = iree_codegen.load_from_buffer %3 : memref<256x256xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<256x256xf16>
      %8 = tensor.empty() : tensor<256x256xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %10 = linalg.matmul {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 128], subgroup_basis = [[2, 2, 1], [0, 1, 2]], workgroup = [64, 64, 0]}>} ins(%6, %7 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%9 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %11 = tensor.empty() : tensor<2x16x8x4x4x4x4xf32>
      %12 = iree_linalg_ext.map_store %10 into %11 {
      ^bb0(%arg0: index, %arg1: index):
        %13:2 = affine.delinearize_index %arg0 into (2, 128) : index, index
        %14:2 = affine.delinearize_index %arg1 into (16, 16) : index, index
        %15:3 = affine.delinearize_index %13#1 into (4, 8, 4) : index, index, index
        %16:2 = affine.delinearize_index %14#1 into (4, 4) : index, index
        iree_linalg_ext.yield %13#0, %14#0, %15#1, %16#1, %15#0, %15#2, %16#0, %true : index, index, index, index, index, index, index, i1
      } : tensor<256x256xf32> into tensor<2x16x8x4x4x4x4xf32> -> tensor<2x16x8x4x4x4x4xf32>
      iree_codegen.store_to_buffer %12, %5 : tensor<2x16x8x4x4x4x4xf32> into memref<2x16x8x4x4x4x4xf32, #amdgpu.address_space<fat_raw_buffer>>
      return
    }
  }
}
}
