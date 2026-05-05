// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_AFFINEEXPRUTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_AFFINEEXPRUTILS_H_

#include <cstdint>
#include <optional>

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Value.h"

namespace mlir::iree_compiler {

/// Returns the constant value represented by an add-only affine expression tree.
std::optional<int64_t> getConstantAffineExprValue(AffineExpr expr);

/// Returns the constant offset if `expr` is exactly `dimPosition + constant`,
/// where the constant is represented through add-only affine expression nodes.
std::optional<int64_t> getUnitDimPlusConstantOffset(AffineExpr expr,
                                                    int64_t dimPosition);

/// Returns the constant offset if `outputIndex` is the same SSA value as
/// `inputIndex`, or if it is a single-result affine.apply whose result is
/// `inputIndex + constant`. The input must appear exactly once as a dimension
/// operand; symbol operands, chained applies, and non-add affine expressions
/// return std::nullopt.
std::optional<int64_t> getConstantUnitOffset(Value outputIndex,
                                             Value inputIndex);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_AFFINEEXPRUTILS_H_
