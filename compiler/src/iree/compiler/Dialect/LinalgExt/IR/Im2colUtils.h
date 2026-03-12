// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: Move Im2colUtils.{h,cpp} from IR/ to Utils/. These are transformation
// helpers consumed by Transforms/ and IR/AggregatedOpInterfaceImpl, not op
// definitions or dialect infrastructure.

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Holds the computed source indices for an im2col operation at a given
/// output position. These indices describe where to read from the input tensor.
struct Im2colSourceIndices {
  /// Full set of offsets into the input tensor, one per input dimension.
  /// When padding is present, these are in the padded coordinate space.
  SmallVector<OpFoldResult> sliceOffsets;
  /// Sizes for each input dimension (1 except for the vectorized dim).
  SmallVector<OpFoldResult> sliceSizes;
};

/// Compute source (input tensor) indices for a given im2col output position.
///
/// Given the loop induction variables representing the current output position,
/// compute the corresponding offsets and sizes into the input tensor. Uses the
/// unified offsets + output_sizes attributes to delinearize each output dim
/// independently, then maps to input coordinates via strides, dilations, and
/// input_k_perm.
///
/// Shared by both the decomposition and vectorization paths.
Im2colSourceIndices computeIm2colSourceIndices(OpBuilder &b, Location loc,
                                                Im2colOp im2colOp,
                                                ArrayRef<Value> ivs,
                                                OpFoldResult innerTileSize);

/// Choose which output dimension to vectorize for an im2col op.
/// Returns the output dimension index, or std::nullopt if no dimension can be
/// vectorized (in which case scalar unrolling should be used).
///
/// \p offsets are the per-output-dim offsets from the im2col op's attributes.
/// For K output dims, the offset of the specific dim being considered is used
/// directly for the contiguity check (no linearization needed).
std::optional<int64_t>
chooseDimToVectorize(OpBuilder &b, Location loc, Im2colOp im2colOp,
                     ArrayRef<Range> iterationDomain,
                     ArrayRef<OpFoldResult> inputSizes,
                     ArrayRef<OpFoldResult> offsets);

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_IR_IM2COLUTILS_H_
