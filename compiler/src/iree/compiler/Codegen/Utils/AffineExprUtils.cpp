// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/AffineExprUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/Support/CheckedArithmetic.h"

namespace mlir::iree_compiler {

std::optional<int64_t> getConstantAffineExprValue(AffineExpr expr) {
  if (auto constantExpr = dyn_cast<AffineConstantExpr>(expr)) {
    return constantExpr.getValue();
  }
  AffineBinaryOpExpr binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binaryExpr || binaryExpr.getKind() != AffineExprKind::Add) {
    return std::nullopt;
  }
  std::optional<int64_t> lhs = getConstantAffineExprValue(binaryExpr.getLHS());
  std::optional<int64_t> rhs = getConstantAffineExprValue(binaryExpr.getRHS());
  if (!lhs || !rhs) {
    return std::nullopt;
  }
  return llvm::checkedAdd(*lhs, *rhs);
}

std::optional<int64_t> getUnitDimPlusConstantOffset(AffineExpr expr,
                                                    int64_t dimPosition) {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    if (static_cast<int64_t>(dimExpr.getPosition()) == dimPosition) {
      return 0;
    }
    return std::nullopt;
  }
  AffineBinaryOpExpr binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binaryExpr || binaryExpr.getKind() != AffineExprKind::Add) {
    return std::nullopt;
  }
  if (std::optional<int64_t> lhsOffset =
          getUnitDimPlusConstantOffset(binaryExpr.getLHS(), dimPosition)) {
    if (std::optional<int64_t> rhsConstant =
            getConstantAffineExprValue(binaryExpr.getRHS())) {
      return llvm::checkedAdd(*lhsOffset, *rhsConstant);
    }
  }
  if (std::optional<int64_t> rhsOffset =
          getUnitDimPlusConstantOffset(binaryExpr.getRHS(), dimPosition)) {
    if (std::optional<int64_t> lhsConstant =
            getConstantAffineExprValue(binaryExpr.getLHS())) {
      return llvm::checkedAdd(*rhsOffset, *lhsConstant);
    }
  }
  return std::nullopt;
}

std::optional<int64_t> getConstantUnitOffset(Value outputIndex,
                                             Value inputIndex) {
  if (outputIndex == inputIndex) {
    return 0;
  }
  affine::AffineApplyOp applyOp =
      outputIndex.getDefiningOp<affine::AffineApplyOp>();
  if (!applyOp) {
    return std::nullopt;
  }
  AffineMap affineMap = applyOp.getAffineMap();
  if (affineMap.getNumResults() != 1) {
    return std::nullopt;
  }

  std::optional<int64_t> inputOperandPosition;
  for (auto [position, operand] : llvm::enumerate(applyOp.getMapOperands())) {
    if (operand != inputIndex) {
      continue;
    }
    if (inputOperandPosition) {
      return std::nullopt;
    }
    inputOperandPosition = static_cast<int64_t>(position);
  }
  if (!inputOperandPosition) {
    return std::nullopt;
  }
  if (*inputOperandPosition >= affineMap.getNumDims()) {
    return std::nullopt;
  }
  return getUnitDimPlusConstantOffset(affineMap.getResult(0),
                                      *inputOperandPosition);
}

} // namespace mlir::iree_compiler
