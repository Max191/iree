// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_BUFFERIZEDISPATCHTENSORLOADSTOREPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

static void
bufferizeFlowDispatchTensorLoad(IRRewriter &rewriter,
                                IREE::Flow::DispatchTensorLoadOp loadOp) {
  Value source = loadOp.getSource();
  auto subspanOp = source.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (!subspanOp) {
    return;
  }
  Value subspanMemref = findOrCreateSubspanBuffer(rewriter, subspanOp);

  rewriter.setInsertionPoint(loadOp);
  Location loc = loadOp->getLoc();
  bool readOnly =
      loadOp.getSourceType().getAccess() == IREE::Flow::TensorAccess::ReadOnly;
  if (equalTensorShape(loadOp.getType(), loadOp.sizes(),
                       loadOp.getSourceType(),
                       loadOp.getSourceDims())) {
    rewriter.replaceOpWithNewOp<IREE::Codegen::LoadFromMemrefOp>(
        loadOp, loadOp.getType(), subspanMemref, readOnly);
    return;
  }
  MemRefType subviewMemRefType = memref::SubViewOp::inferRankReducedResultType(
      loadOp.getType().getShape(), cast<MemRefType>(subspanMemref.getType()),
      loadOp.getMixedOffsets(), loadOp.getMixedSizes(),
      loadOp.getMixedStrides());
  Value subview = rewriter.create<memref::SubViewOp>(
      loc, llvm::cast<MemRefType>(subviewMemRefType), subspanMemref,
      loadOp.getMixedOffsets(), loadOp.getMixedSizes(),
      loadOp.getMixedStrides()).getResult();
  rewriter.replaceOpWithNewOp<IREE::Codegen::LoadFromMemrefOp>(
      loadOp, loadOp.getType(), subview, readOnly);
}

static void
bufferizeFlowDispatchTensorStore(IRRewriter &rewriter,
                                 IREE::Flow::DispatchTensorStoreOp storeOp) {
  Value target = storeOp.getTarget();
  auto subspanOp = target.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (!subspanOp) {
    return;
  }
  Value subspanMemref = findOrCreateSubspanBuffer(rewriter, subspanOp);

  rewriter.setInsertionPoint(storeOp);
  Location loc = storeOp->getLoc();
  Value value = storeOp.getValue();
  if (equalTensorShape(storeOp.getValueType(), storeOp.getSizes(),
            storeOp.getTargetType(), storeOp.getTargetDims())) {
    rewriter.replaceOpWithNewOp<IREE::Codegen::StoreToMemrefOp>(
        storeOp, value, subspanMemref);
    return;
  }
  auto subviewMemRefType = llvm::cast<MemRefType>(
      memref::SubViewOp::inferRankReducedResultType(
          cast<ShapedType>(value.getType()).getShape(),
          cast<MemRefType>(subspanMemref.getType()), storeOp.getMixedOffsets(),
          storeOp.getMixedSizes(), storeOp.getMixedStrides()));
  Value subview = rewriter.create<memref::SubViewOp>(
      loc, subviewMemRefType, subspanMemref,
      storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
      storeOp.getMixedStrides()).getResult();
  rewriter.replaceOpWithNewOp<IREE::Codegen::StoreToMemrefOp>(
      storeOp, value, subview);
}

namespace {

struct BufferizeDispatchTensorLoadStorePass final
    : impl::BufferizeDispatchTensorLoadStorePassBase<BufferizeDispatchTensorLoadStorePass> {
  using BufferizeDispatchTensorLoadStorePassBase::BufferizeDispatchTensorLoadStorePassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();
    SmallVector<IREE::Flow::DispatchTensorLoadOp> loadOps;
    SmallVector<IREE::Flow::DispatchTensorStoreOp> storeOps;
    funcOp.walk([&](Operation *op) {
      if (auto loadOp = dyn_cast<IREE::Flow::DispatchTensorLoadOp>(op)) {
        loadOps.push_back(loadOp);
      }
      if (auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(op)) {
        storeOps.push_back(storeOp);
      }
    });

    IRRewriter rewriter(context);
    for (IREE::Flow::DispatchTensorLoadOp loadOp : loadOps) {
      bufferizeFlowDispatchTensorLoad(rewriter, loadOp);
    }
    for (IREE::Flow::DispatchTensorStoreOp storeOp : storeOps) {
      bufferizeFlowDispatchTensorStore(rewriter, storeOp);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
