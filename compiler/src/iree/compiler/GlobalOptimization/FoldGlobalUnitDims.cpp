// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

//===----------------------------------------------------------------------===//
// Util functions
//===----------------------------------------------------------------------===//

static LogicalResult foldUnitDimsOnGlobal(IRRewriter &rewriter, IREE::Util::GlobalOp global, SmallVector<Operation *> loadStoreOps, SymbolTable moduleSymbols) {
  // Create a new transformed GlobalOp.
  SmallVector<int64_t> newShape;
  auto globalType = cast<RankedTensorType>(global.getType());
  for (auto size : globalType.getShape()) {
    if (size != 1) {
      newShape.push_back(size);
    }
  }
  auto newGlobalType = RankedTensorType::get(newShape, globalType.getElementType());
  std::optional<TypedAttr> newInitialValue = std::nullopt;
  if (auto initialValue = global.getInitialValue()) {
    if (auto uninitializedAttr = dyn_cast<IREE::Util::UninitializedAttr>(initialValue.value())) {
      newInitialValue = IREE::Util::UninitializedAttr::get(rewriter.getContext(), newGlobalType);
    } else {
      return failure();
    }
  }
  StringRef newGlobalName(global.getGlobalName());
  rewriter.setInsertionPoint(global);
  auto newGlobal =
      rewriter.create<IREE::Util::GlobalOp>(global->getLoc(), newGlobalName,
                                global.getIsMutable(), newGlobalType, newInitialValue);
  moduleSymbols.insert(newGlobal);
  SymbolTable::setSymbolVisibility(newGlobal,
                                   SymbolTable::getSymbolVisibility(global));

  // Rewrite loads and stores to use the new global.
  auto expandShapeReInds = getReassociationIndicesForReshape(globalType, newGlobalType);
  if (!expandShapeReInds) {
    return failure();
  }
  for (auto loadOrStore : loadStoreOps) {
    rewriter.setInsertionPoint(loadOrStore);
    if (auto load = dyn_cast<IREE::Util::GlobalLoadOp>(loadOrStore)) {
      auto newLoad = rewriter.create<IREE::Util::GlobalLoadOp>(load->getLoc(), newGlobal);
      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(load, globalType, newLoad, expandShapeReInds.value());
    } else if (auto store = dyn_cast<IREE::Util::GlobalStoreOp>(loadOrStore)) {
      auto collapse = rewriter.create<tensor::CollapseShapeOp>(store.getLoc(), newGlobalType, store.getOperand(), expandShapeReInds.value());
      rewriter.create<IREE::Util::GlobalStoreOp>(store->getLoc(), collapse, newGlobal);
      rewriter.eraseOp(store);
    } else {
      return failure();
    }
  }
  rewriter.eraseOp(global);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

class FoldGlobalUnitDimsPass
    : public FoldGlobalUnitDimsBase<
          FoldGlobalUnitDimsPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable moduleSymbols(moduleOp);
    Explorer explorer(moduleOp, TraversalAction::RECURSE);
    DenseMap<StringRef, SmallVector<Operation *>> loadStoreMap;
    auto addToLoadStoreMap = [&](StringRef name, Operation *loadStoreOp) {
      if (loadStoreMap.contains(name)) {
        loadStoreMap[name].push_back(loadStoreOp);
      } else {
        SmallVector<Operation *> loadStores(1, loadStoreOp);
        loadStoreMap.insert(std::make_pair(name, loadStores));
      }
    };
    auto walkFn = [&](Operation *op) -> WalkResult {
      if (auto load = dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
        addToLoadStoreMap(load.getGlobalName(), op);
      } else if (auto store = dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
        addToLoadStoreMap(store.getGlobalName(), op);
      }
      return WalkResult::advance();
    };
    explorer.walk(walkFn);
    IRRewriter rewriter(&getContext());
    SmallVector<IREE::Util::GlobalOp> foldableGlobals;
    for (auto global : moduleOp.getOps<IREE::Util::GlobalOp>()) {
      if (global.getIsMutable()) {
        if (auto tensorType = dyn_cast<RankedTensorType>(global.getType())) {
          if (llvm::any_of(tensorType.getShape(), [](int64_t size){ return size == 1; } )) {
            foldableGlobals.push_back(global);
          }
        }
      }
    }
    for (auto global : foldableGlobals) {
      if (failed(foldUnitDimsOnGlobal(rewriter, global, loadStoreMap[global.getGlobalName()], moduleSymbols))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createFoldGlobalUnitDimsPass() {
  return std::make_unique<FoldGlobalUnitDimsPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
