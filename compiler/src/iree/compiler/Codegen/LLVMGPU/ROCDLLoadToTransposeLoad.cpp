// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-rocdl-load-to-transpose-load"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLLOADTOTRANSPOSELOADPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {

/// Attempts to lower a vector.transfer_read into amdgpu.transpose_load.
/// If the transformation is not possible or not profitable, does nothing.
static void lowerTransferReadToTransposeLoad(vector::TransferReadOp transferOp,
                                             IRRewriter &rewriter) {
  // TODO: Implement the actual transformation logic here.
  // This should:
  // 1. Check if indices come from transpose_load_index_hint
  // 2. Verify memory space is LDS (workgroup memory)
  // 3. Check access pattern matches transpose_load requirements
  // 4. Validate vector size and element type
  // 5. Create amdgpu.transpose_load op if all checks pass
  //
  // For now, do nothing (no transformation applied).
  LLVM_DEBUG(llvm::dbgs() << "Analyzing transfer_read for transpose_load: "
                          << transferOp << "\n");
}

/// Pass to lower vector.transfer_read operations to amdgpu.transpose_load
/// operations when profitable, based on iree_gpu.transpose_load_index_hint
/// annotations.
struct ROCDLLoadToTransposeLoadPass final
    : impl::ROCDLLoadToTransposeLoadPassBase<ROCDLLoadToTransposeLoadPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    // Step 1: Collect all vector.transfer_read operations
    SmallVector<vector::TransferReadOp> transferReads;
    funcOp.walk([&](vector::TransferReadOp transferOp) {
      transferReads.push_back(transferOp);
    });

    // Step 2: Attempt to transform each transfer_read
    for (auto transferOp : transferReads) {
      rewriter.setInsertionPoint(transferOp);
      lowerTransferReadToTransposeLoad(transferOp, rewriter);
      // Note: For now, this does nothing (stubbed)
    }

    // Step 3: Remove all remaining transpose_load_index_hint operations
    // These hints are purely for optimization and safe to remove if not used
    SmallVector<IREE::GPU::TransposeLoadIndexHintOp> hintOps;
    funcOp.walk([&](IREE::GPU::TransposeLoadIndexHintOp hintOp) {
      hintOps.push_back(hintOp);
    });

    for (auto hintOp : hintOps) {
      // Replace hint results with the original operands (pass-through)
      for (auto [result, operand] :
           llvm::zip(hintOp.getResults(), hintOp.getOperands())) {
        result.replaceAllUsesWith(operand);
      }
      rewriter.eraseOp(hintOp);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
