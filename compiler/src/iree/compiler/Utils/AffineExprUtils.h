// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_AFFINEEXPRUTILS_H_
#define IREE_COMPILER_UTILS_AFFINEEXPRUTILS_H_

#include "mlir/IR/AffineExpr.h"

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

/// Returns true if `expr` references any affine dim. This is the negation of
/// MLIR's symbolic-or-constant affine expression classification.
inline bool affineExprUsesDim(AffineExpr expr) {
  return !expr.isSymbolicOrConstant();
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_AFFINEEXPRUTILS_H_
