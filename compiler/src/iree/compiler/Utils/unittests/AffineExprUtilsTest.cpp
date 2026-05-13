// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include "iree/compiler/Utils/AffineExprUtils.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::iree_compiler;

TEST(AffineExprUsesSymbol, FindsSymbolsRecursively) {
  MLIRContext context;
  OpBuilder builder(&context);
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr s0 = builder.getAffineSymbolExpr(0);
  AffineExpr c0 = builder.getAffineConstantExpr(0);

  EXPECT_FALSE(affineExprUsesSymbol(c0));
  EXPECT_FALSE(affineExprUsesSymbol(d0));
  EXPECT_TRUE(affineExprUsesSymbol(s0));
  EXPECT_TRUE(affineExprUsesSymbol(d0 + 4 * s0));
}

TEST(AffineExprUsesDim, FindsDimsRecursively) {
  MLIRContext context;
  OpBuilder builder(&context);
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr s0 = builder.getAffineSymbolExpr(0);
  AffineExpr c0 = builder.getAffineConstantExpr(0);

  EXPECT_FALSE(affineExprUsesDim(c0));
  EXPECT_TRUE(affineExprUsesDim(d0));
  EXPECT_FALSE(affineExprUsesDim(s0));
  EXPECT_FALSE(affineExprUsesDim(s0 + 3));
  EXPECT_TRUE(affineExprUsesDim((s0 + 3).floorDiv(2) + d0));
}
