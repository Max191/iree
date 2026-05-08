// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_AFFINEEXPRUTILS_H_
#define IREE_COMPILER_UTILS_AFFINEEXPRUTILS_H_

#include "llvm/Support/CheckedArithmetic.h"
#include "mlir/IR/AffineExpr.h"

#include <optional>

namespace mlir::iree_compiler {

/// Returns true if `expr` references any affine symbol.
inline bool affineExprUsesSymbol(AffineExpr expr) {
  if (isa<AffineSymbolExpr>(expr)) {
    return true;
  }
  auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  return binaryExpr && (affineExprUsesSymbol(binaryExpr.getLHS()) ||
                        affineExprUsesSymbol(binaryExpr.getRHS()));
}

/// Returns the linear coefficient of `dimPosition` in `expr`.
///
/// This intentionally handles only the affine fragment needed by current
/// transfer gather/scatter lowering: dims, symbols, constants, additions, and
/// multiplication by constants. Unsupported expressions return nullopt.
inline std::optional<int64_t>
getAffineDimCoefficient(AffineExpr expr, unsigned dimPosition) {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    return dimExpr.getPosition() == dimPosition ? 1 : 0;
  }
  if (isa<AffineSymbolExpr, AffineConstantExpr>(expr)) {
    return 0;
  }
  auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binaryExpr) {
    return std::nullopt;
  }
  switch (binaryExpr.getKind()) {
  case AffineExprKind::Add: {
    std::optional<int64_t> lhs =
        getAffineDimCoefficient(binaryExpr.getLHS(), dimPosition);
    std::optional<int64_t> rhs =
        getAffineDimCoefficient(binaryExpr.getRHS(), dimPosition);
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    return llvm::checkedAdd(*lhs, *rhs);
  }
  case AffineExprKind::Mul: {
    auto lhsConstant = dyn_cast<AffineConstantExpr>(binaryExpr.getLHS());
    auto rhsConstant = dyn_cast<AffineConstantExpr>(binaryExpr.getRHS());
    if (lhsConstant) {
      std::optional<int64_t> rhs =
          getAffineDimCoefficient(binaryExpr.getRHS(), dimPosition);
      return rhs ? llvm::checkedMul(*rhs, lhsConstant.getValue())
                 : std::nullopt;
    }
    if (rhsConstant) {
      std::optional<int64_t> lhs =
          getAffineDimCoefficient(binaryExpr.getLHS(), dimPosition);
      return lhs ? llvm::checkedMul(*lhs, rhsConstant.getValue())
                 : std::nullopt;
    }
    return std::nullopt;
  }
  default:
    return std::nullopt;
  }
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_AFFINEEXPRUTILS_H_
