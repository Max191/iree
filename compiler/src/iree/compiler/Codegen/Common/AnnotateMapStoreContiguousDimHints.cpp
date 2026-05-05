// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/AffineExprUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ANNOTATEMAPSTORECONTIGUOUSDIMHINTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Infers only exact unit-contiguous dimensions that this pass can certify
/// syntactically: the yielded output index must be the input block argument, or
/// a single-result affine.apply using that block argument once as `d + C` after
/// add-only constant folding. Ambiguous input dims that match zero or multiple
/// output dims are intentionally left unhinted; vectorization may still use its
/// legacy innermost fallback.
static SmallVector<int64_t>
inferContiguousDimHints(IREE::LinalgExt::MapStoreOp mapStoreOp) {
  SmallVector<SmallVector<int64_t>> candidateOutputDims(
      mapStoreOp.getInputRank());
  for (int64_t outputDim = 0, e = mapStoreOp.getOutputRank(); outputDim < e;
       ++outputDim) {
    Value outputIndex = mapStoreOp.getOutputIndex(outputDim);
    for (int64_t inputDim = 0, f = mapStoreOp.getInputRank(); inputDim < f;
         ++inputDim) {
      std::optional<int64_t> offset = getConstantUnitOffset(
          outputIndex, mapStoreOp.getInputIndex(inputDim));
      if (offset.has_value()) {
        candidateOutputDims[inputDim].push_back(outputDim);
      }
    }
  }

  SmallVector<int64_t> hints;
  for (auto [inputDim, outputDims] : llvm::enumerate(candidateOutputDims)) {
    if (outputDims.size() != 1) {
      continue;
    }
    hints.push_back(inputDim);
    hints.push_back(outputDims.front());
  }
  return hints;
}

struct AnnotateMapStoreContiguousDimHintsPass final
    : impl::AnnotateMapStoreContiguousDimHintsPassBase<
          AnnotateMapStoreContiguousDimHintsPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    funcOp.walk([&](IREE::LinalgExt::MapStoreOp mapStoreOp) {
      if (mapStoreOp.getContiguousDimHintsAttr()) {
        return;
      }
      SmallVector<int64_t> hints = inferContiguousDimHints(mapStoreOp);
      if (hints.empty()) {
        return;
      }
      mapStoreOp.setContiguousDimHintsAttr(
          DenseI64ArrayAttr::get(mapStoreOp.getContext(), hints));
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler
