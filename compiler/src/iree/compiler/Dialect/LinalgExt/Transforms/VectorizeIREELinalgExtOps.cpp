// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <functional>
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_VECTORIZEIREELINALGEXTOPSPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct VectorizeStaticMapScatterOpPattern final
    : OpRewritePattern<IREE::LinalgExt::MapScatterOp> {
  using OpRewritePattern<IREE::LinalgExt::MapScatterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::LinalgExt::MapScatterOp mapScatterOp,
                                PatternRewriter &rewriter) const override {
    ShapedType inputType = mapScatterOp.getInputType();
    if (isa<VectorType>(inputType)) {
      return rewriter.notifyMatchFailure(mapScatterOp,
                                         "map_scatter is already vectorized");
    }
    if (!inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(mapScatterOp,
                                         "map_scatter has non-static shape");
    }
    Location loc = mapScatterOp.getLoc();
    rewriter.setInsertionPoint(mapScatterOp);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> zeros(inputType.getRank(), zero);
    auto inputVectorType =
        VectorType::get(inputType.getShape(), inputType.getElementType());
    Value inputVector = rewriter.create<vector::TransferReadOp>(
        loc, inputVectorType, mapScatterOp.getInput(), /*indices=*/zeros);

    // // Create a loop nest to transfer write each element of the input vector
    // // into the output of the map_scatter op.
    // bool shouldYieldValue = mapScatterOp.hasPureTensorSemantics();
    // auto bodyBuilder = [&](OpBuilder &b, Location loopLoc, ValueRange ivs,
    //                        ValueRange iterArgs) -> SmallVector<Value> {
    //   SmallVector<Value> resultsToYield;
    //   auto inlineBodyBuilder = [&](OpBuilder nestedBuilder, Location
    //   nestedLoc,
    //                                ArrayRef<Value> yieldedValues) {
    //     Value ifCond = yieldedValues.back();
    //     ArrayRef<Value> storeIndices =
    //         yieldedValues.take_front(yieldedValues.size() - 1);
    //     auto ifOp = nestedBuilder.create<scf::IfOp>(
    //         nestedLoc, mapScatterOp.getResultTypes(), ifCond,
    //         /*withElseRegion=*/shouldYieldValue);
    //     Block &thenBlock = ifOp.getThenRegion().front();
    //     nestedBuilder.setInsertionPointToStart(&thenBlock);
    //     auto inputElementType =
    //         VectorType::get({1}, inputType.getElementType());
    //     SmallVector<int64_t> staticIndices(ivs.size(), ShapedType::kDynamic);
    //     Value inputElement = nestedBuilder.create<vector::ExtractOp>(
    //         nestedLoc, inputElementType, inputVector, ivs,
    //         nestedBuilder.getDenseI64ArrayAttr(staticIndices));
    //     auto transferWriteOp = nestedBuilder.create<vector::TransferWriteOp>(
    //         nestedLoc, inputElement, mapScatterOp.getOutput(), storeIndices);
    //     if (shouldYieldValue) {
    //       nestedBuilder.create<scf::YieldOp>(nestedLoc,
    //                                          transferWriteOp.getResults());
    //     }
    //     // Only create a then block if a Value must be yielded.
    //     if (!shouldYieldValue) {
    //       return;
    //     }
    //     Block &elseBlock = ifOp.getElseRegion().front();
    //     nestedBuilder.setInsertionPointToStart(&elseBlock);
    //     nestedBuilder.create<scf::YieldOp>(nestedLoc, iterArgs.front());
    //     resultsToYield.push_back(ifOp.getResult(0));
    //   };
    //   mapScatterOp.inlineMapScatterBody(b, loopLoc, ivs, inlineBodyBuilder);
    //   return resultsToYield;
    // };
    // SmallVector<Value> ubs =
    //     llvm::map_to_vector(inputType.getShape(), [&](int64_t size) -> Value
    //     {
    //       return rewriter.create<arith::ConstantIndexOp>(loc, size);
    //     });
    // Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    // SmallVector<Value> ones(inputType.getRank(), one);
    // SmallVector<Value> iterArgs;
    // if (mapScatterOp.hasPureTensorSemantics()) {
    //   iterArgs.push_back(mapScatterOp.getOutput());
    // }
    // scf::LoopNest loopNest =
    //     scf::buildLoopNest(rewriter, loc, /*lbs=*/zeros, ubs, /*steps=*/ones,
    //                        iterArgs, bodyBuilder);
    auto vectorizedMapScatterOp =
        clone(rewriter, mapScatterOp, mapScatterOp.getResultTypes(),
              {inputVector, mapScatterOp.getOutput()});
    rewriter.replaceOp(mapScatterOp, vectorizedMapScatterOp);
    // rewriter.replaceOp(mapScatterOp, loopNest.loops.front());
    return success();
  }
};

struct VectorizeIREELinalgExtOpsPass final
    : impl::VectorizeIREELinalgExtOpsPassBase<VectorizeIREELinalgExtOpsPass> {
  void runOnOperation() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<VectorizeStaticMapScatterOpPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::LinalgExt
