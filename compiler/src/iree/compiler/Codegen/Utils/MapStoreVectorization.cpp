// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/MapStoreVectorization.h"

namespace mlir::iree_compiler {

DenseI64ArrayAttr
getMapStoreContiguousDimHints(IREE::LinalgExt::MapStoreOp mapStoreOp) {
  return mapStoreOp->getAttrOfType<DenseI64ArrayAttr>(
      kMapStoreContiguousDimHintsAttrName);
}

AffineMapAttr getMapStoreTransferScatterIndexingMap(
    IREE::LinalgExt::MapStoreOp mapStoreOp) {
  return mapStoreOp->getAttrOfType<AffineMapAttr>(
      kMapStoreTransferScatterIndexingMapAttrName);
}

void setMapStoreTransferScatterIndexingMap(
    IREE::LinalgExt::MapStoreOp mapStoreOp, AffineMapAttr indexingMapAttr) {
  mapStoreOp->setAttr(kMapStoreTransferScatterIndexingMapAttrName,
                      indexingMapAttr);
}

bool hasMapStoreVectorizationHints(IREE::LinalgExt::MapStoreOp mapStoreOp) {
  return getMapStoreContiguousDimHints(mapStoreOp) ||
         getMapStoreTransferScatterIndexingMap(mapStoreOp);
}

} // namespace mlir::iree_compiler
