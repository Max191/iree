// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::iree_compiler {

std::optional<OpPassManager>
getFunctionOpInterfacePassManager(FunctionOpInterface interfaceOp) {
  return TypeSwitch<Operation *, std::optional<OpPassManager>>(
             interfaceOp.getOperation())
      .Case<func::FuncOp, IREE::Util::FuncOp>(
          [&](auto funcOp) { return OpPassManager(funcOp.getOperationName()); })
      .Default([&](Operation *op) { return std::nullopt; });
}

static Operation *skipCastsDefiningOp(Value v) {
  auto producer = v.getDefiningOp();
  while (producer) {
    auto castProducer = dyn_cast<tensor::CastOp>(producer);
    if (!castProducer) {
      break;
    }
    producer = castProducer.getSource().getDefiningOp();
  }
  return producer;
}

void ConfigTrackingListener::notifyOperationReplaced(Operation *op,
                                                     ValueRange replacement) {
  // We have no way to track replacements without a producer.
  if (replacement.empty()) {
    return;
  }

  IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
      getLoweringConfig(op);
  if (!loweringConfig) {
    return;
  }

  // Must have a producer of the same type to track the lowering config.
  auto producer = skipCastsDefiningOp(replacement.front());
  if (!producer || producer->getName() != op->getName()) {
    return;
  }

  for (Value v : replacement.drop_front()) {
    // Conservatively require that all replacements are produced by the same
    // operation.
    if (skipCastsDefiningOp(v) != producer) {
      return;
    }
  }

  // No need to add the lowering config if it's already present.
  if (getLoweringConfig(producer)) {
    return;
  }

  setLoweringConfig(producer, loweringConfig);
}

void configureCodegenGreedyRewrite(GreedyRewriteConfig &config) {
  // Inherit the same config defaults from the upstream canonicalizer pass.
  config.setUseTopDownTraversal().setRegionSimplificationLevel(
      GreedySimplifyRegionLevel::Normal);
}

void populateCodegenCanonicalizationPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  for (auto *dialect : context->getLoadedDialects()) {
    dialect->getCanonicalizationPatterns(patterns);
  }
  for (RegisteredOperationName op : context->getRegisteredOperations()) {
    op.getCanonicalizationPatterns(patterns, context);
  }
}

LogicalResult applyPatternsWithConfigTracking(
    Operation *op, const FrozenRewritePatternSet &patterns,
    GreedyRewriteConfig &config, bool testConvergence,
    StringRef convergenceError) {
  ConfigTrackingListener listener;
  config.setListener(&listener);
  LogicalResult didConverge = applyPatternsGreedily(op, patterns, config);
  config.setListener(nullptr);
  if (testConvergence && failed(didConverge)) {
    op->emitError(convergenceError);
    return failure();
  }
  return success();
}

} // namespace mlir::iree_compiler
