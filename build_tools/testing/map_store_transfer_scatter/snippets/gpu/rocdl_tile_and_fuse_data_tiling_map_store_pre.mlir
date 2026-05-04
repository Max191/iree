// Extracted from compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_tile_and_fuse.mlir before GPU vector distribution.

#translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [64, 1, 1] subgroup_size = 64>
#lowering_config = #iree_gpu.lowering_config<{
  thread = [1, 4], workgroup = [1, 256]
}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
]>

hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @small_elementwise ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @map_store() attributes {translation_info = #translation_info} {
        %c0 = arith.constant 0 : index
        %true = arith.constant true
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<2048x2048xf32, #hal.descriptor_type<storage_buffer>>
        %4 = amdgpu.fat_raw_buffer_cast %3 resetOffset : memref<2048x2048xf32, #hal.descriptor_type<storage_buffer>> to memref<2048x2048xf32, #amdgpu.address_space<fat_raw_buffer>>
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : memref<16x16x128x128xf32, #hal.descriptor_type<storage_buffer>>
        %6 = amdgpu.fat_raw_buffer_cast %5 resetOffset : memref<16x16x128x128xf32, #hal.descriptor_type<storage_buffer>> to memref<16x16x128x128xf32, #amdgpu.address_space<fat_raw_buffer>>
        %7 = iree_codegen.load_from_buffer %4 : memref<2048x2048xf32, #amdgpu.address_space<fat_raw_buffer>> -> tensor<2048x2048xf32>
        %8 = tensor.empty() : tensor<16x16x128x128xf32>
        %9 = iree_linalg_ext.map_store {lowering_config = #lowering_config} %7 into %8 {
        ^bb0(%arg0: index, %arg1: index):
          %10:2 = affine.delinearize_index %arg0 into (16, 128) : index, index
          %11:2 = affine.delinearize_index %arg1 into (16, 128) : index, index
          iree_linalg_ext.yield %10#0, %11#0, %10#1, %11#1, %true : index, index, index, index, i1
        } : tensor<2048x2048xf32> into tensor<16x16x128x128xf32> -> tensor<16x16x128x128xf32>
        iree_codegen.store_to_buffer %9, %6 : tensor<16x16x128x128xf32> into memref<16x16x128x128xf32, #amdgpu.address_space<fat_raw_buffer>>
        return
      }
    }
  }
}
