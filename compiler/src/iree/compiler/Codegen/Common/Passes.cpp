// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {

llvm::cl::opt<std::string> clCodegenPreprocessingTransformDialectFileName(
    "iree-codegen-use-preprocessing-transform-dialect-pass",
    llvm::cl::desc(
        "MLIR file containing a transform dialect specification to apply"),
    llvm::cl::init(""));
llvm::cl::opt<std::string> clCodegenPreprocessingTransformDialectDebugPayloadTag(
    "iree-codegen-preprocessing-transform-dialect-debug-payload-tag",
    llvm::cl::desc("tag attribute value for the transform dialect interpreter "
                   "payload root operation"),
    llvm::cl::init(""));
llvm::cl::opt<std::string> clCodegenPreprocessingTransformDialectDebugTransformTag(
    "iree-codegen-preprocessing-transform-dialect-debug-transform-tag",
    llvm::cl::desc(
        "tag attribute value for the transform dialect transform op container"),
    llvm::cl::init(""));

void addCommonTargetExecutablePreprocessingPasses(OpPassManager &passManager) {
  passManager.addNestedPass<func::FuncOp>(createTypePropagationPass());
  passManager.addPass(createBubbleUpOrdinalOpsPass());
  passManager.addPass(createBufferizeCopyOnlyDispatchesPass());
  passManager.addNestedPass<func::FuncOp>(
      IREE::LinalgExt::createDecomposeSoftmaxPass());
  if (!clCodegenPreprocessingTransformDialectFileName.empty())
      passManager.addPass(createTransformDialectInterpreterPass(
          clCodegenPreprocessingTransformDialectFileName,
          clCodegenPreprocessingTransformDialectDebugPayloadTag,
          clCodegenPreprocessingTransformDialectDebugTransformTag));
}

//===---------------------------------------------------------------------===//
// Register Common Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Common/Passes.h.inc"
} // namespace

void registerCodegenCommonPasses() {
  // Generated.
  registerPasses();
}
} // namespace iree_compiler
} // namespace mlir
