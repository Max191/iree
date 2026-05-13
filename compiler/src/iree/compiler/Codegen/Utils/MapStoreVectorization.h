// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_MAPSTOREVECTORIZATION_H_
#define IREE_COMPILER_CODEGEN_UTILS_MAPSTOREVECTORIZATION_H_

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler {

inline constexpr llvm::StringLiteral kMapStoreContiguousDimHintsAttrName =
    "contiguous_dim_hints";
inline constexpr llvm::StringLiteral
    kMapStoreTransferScatterIndexingMapAttrName =
        "transfer_scatter_indexing_map";

/// Returns the legacy contiguous-dimension vectorization hint, or null if the
/// map_store has no such hint.
DenseI64ArrayAttr
getMapStoreContiguousDimHints(IREE::LinalgExt::MapStoreOp mapStoreOp);

/// Returns the transfer_scatter indexing-map vectorization hint, or null if the
/// map_store has no such hint.
AffineMapAttr getMapStoreTransferScatterIndexingMap(
    IREE::LinalgExt::MapStoreOp mapStoreOp);

/// Sets the transfer_scatter indexing-map vectorization hint.
void setMapStoreTransferScatterIndexingMap(
    IREE::LinalgExt::MapStoreOp mapStoreOp, AffineMapAttr indexingMapAttr);

/// Returns true if either legacy or transfer_scatter vectorization metadata is
/// already attached to the map_store.
bool hasMapStoreVectorizationHints(IREE::LinalgExt::MapStoreOp mapStoreOp);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_MAPSTOREVECTORIZATION_H_
