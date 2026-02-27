// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Choose which output dimension to vectorize for an im2col op.
/// Returns the output dimension index, or std::nullopt if no dimension can be
/// vectorized (in which case scalar unrolling should be used).
std::optional<int64_t>
chooseDimToVectorize(OpBuilder &b, Location loc, Im2colOp im2colOp,
                     ArrayRef<Range> iterationDomain,
                     ArrayRef<OpFoldResult> inputSizes,
                     OpFoldResult kOffset);

/// Compute the linearized K offset for an im2col op by combining
/// per-K-output-dim offsets using their output_sizes as a mixed-radix basis.
/// Returns a single index-typed OpFoldResult.
OpFoldResult linearizeIm2colKOffsets(OpBuilder &b, Location loc,
                                     Im2colOp im2colOp);

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_
