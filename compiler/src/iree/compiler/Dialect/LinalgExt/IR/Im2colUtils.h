// Copyright 2025 The IREE Authors
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
                     SmallVector<Range> iterationDomain,
                     SmallVector<OpFoldResult> inputSizes,
                     OpFoldResult kOffset);

/// Linearize a multi-dimensional index using affine arithmetic.
/// Computes the dot product: sum(inds[i] * basis[i]).
OpFoldResult linearizeIm2colIndex(OpBuilder &b, Location loc,
                                  ArrayRef<OpFoldResult> inds,
                                  ArrayRef<OpFoldResult> basis);

/// Result of computeIm2colMBasis: a basis vector and the hasOuterBound flag
/// to use when creating an AffineDelinearizeIndexOp from this basis.
struct Im2colMBasisResult {
  SmallVector<OpFoldResult> basis;
  bool hasOuterBound;
};

/// Compute the M-basis for delinearizing the linearized M index back into
/// per-M-dimension (input spatial) coordinates.
///
/// For "expanded" M (one M output dim per input spatial dim), returns N-1
/// inner basis elements derived from the op's m_strides attribute, with
/// hasOuterBound=false. This correctly handles:
///   - Forward convolution: m_strides encodes the output spatial sizes (e.g.
///     OH*OW) in the full output space.
///   - Backward-weight convolution with dilation > 1: m_strides encodes KW,
///     the kernel spatial size — not the formula-derived approximation that
///     gives wrong results in the presence of dilation.
///   - Tiled im2col ops: m_strides are unchanged by tiling (unlike the output
///     tensor shape which shrinks to tile dimensions).
///
/// For "flat" M (multiple input spatial dims merged into a single M output
/// dim), falls back to computing output sizes from the input tensor via the
/// convolution output size formula, with hasOuterBound=true. This is only
/// correct for forward convolution, but flat-M im2col ops are only produced
/// for forward conv in practice.
Im2colMBasisResult computeIm2colMBasis(OpBuilder &b, Location loc,
                                       Im2colOp im2colOp,
                                       ArrayRef<OpFoldResult> inputSizes);

/// Compute the K-basis for delinearizing the K index into kernel window
/// offsets and channel offsets. Respects the input_k_perm attribute.
SmallVector<OpFoldResult> computeIm2colKBasis(
    Im2colOp im2colOp, ArrayRef<OpFoldResult> inputSizes);

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_
