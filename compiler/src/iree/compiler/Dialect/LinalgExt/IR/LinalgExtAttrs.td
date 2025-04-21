// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_LINALGEXTATTRS
#define IREE_COMPILER_DIALECT_LINALGEXT_LINALGEXTATTRS

include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtBase.td"
include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.td"
include "mlir/Dialect/Utils/StructuredOpsUtils.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Enum definitions
//===----------------------------------------------------------------------===//

def IREELinalgExt_IteratorTypeEnum : EnumAttr<IREELinalgExt_Dialect,
    IteratorType, "iterator_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def IREELinalgExt_IteratorTypeArrayAttr :
    TypedArrayAttrBase<IREELinalgExt_IteratorTypeEnum,
    "LinalgExt iterator type">;

//===----------------------------------------------------------------------===//
// Index Transformation Attributes
//===----------------------------------------------------------------------===//

def IREELinalgExt_TransposeIndicesAttr : AttrDef<IREELinalgExt_Dialect, "TransposeIndices", [
  DeclareAttrInterfaceMethods<IREELinalgExt_IndexTransformationInterfaceAttr, [
    "getNumResultIndices",
    "getNumInputIndices",
    "getNumDynamicIndices",
    "transformIndices",
  ]>
]> {
  let mnemonic = "transpose_indices";
  let cppNamespace = "::mlir::iree_compiler::IREE::LinalgExt";

  let description = [{
    Attribute describing a transposition of indices.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "permutation">:$permutation
  );

  let assemblyFormat = "custom<DynamicI64ArrayAttr>($permutation)";
  let genVerifyDecl = 1;
}

def IREELinalgExt_LinearizeIndicesAttr : AttrDef<IREELinalgExt_Dialect, "LinearizeIndices", [
  DeclareAttrInterfaceMethods<IREELinalgExt_IndexTransformationInterfaceAttr, [
    "getNumResultIndices",
    "getNumInputIndices",
    "getNumDynamicIndices",
    "transformIndices",
  ]>
]> {
  let mnemonic = "linearize_indices";
  let cppNamespace = "::mlir::iree_compiler::IREE::LinalgExt";

  let description = [{
    Attribute describing a linearization of a set of indices with a given basis.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "basis">:$basis
  );

  let assemblyFormat = "custom<DynamicI64ArrayAttr>($basis)";
}

def IREELinalgExt_DelinearizeIndicesAttr : AttrDef<IREELinalgExt_Dialect, "DelinearizeIndices", [
  DeclareAttrInterfaceMethods<IREELinalgExt_IndexTransformationInterfaceAttr, [
    "getNumResultIndices",
    "getNumInputIndices",
    "getNumDynamicIndices",
    "transformIndices",
  ]>
]> {
  let mnemonic = "delinearize_indices";
  let cppNamespace = "::mlir::iree_compiler::IREE::LinalgExt";

  let description = [{
    Attribute describing a delinearization of an index with a given basis.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "basis">:$basis
  );

  let assemblyFormat = "custom<DynamicI64ArrayAttr>($basis)";
}

def IREELinalgExt_ClampIndicesAttr : AttrDef<IREELinalgExt_Dialect, "ClampIndices", [
  DeclareAttrInterfaceMethods<IREELinalgExt_IndexTransformationInterfaceAttr, [
    "getNumResultIndices",
    "getNumInputIndices",
    "getNumDynamicIndices",
    "transformIndices",
  ]>
]> {
  let mnemonic = "clamp_indices";
  let cppNamespace = "::mlir::iree_compiler::IREE::LinalgExt";

  let description = [{
    Attribute describing upper bounds to clamp a set of indices.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "bounds">:$bounds
  );

  let assemblyFormat = "custom<DynamicI64ArrayAttr>($bounds)";
}

def IREELinalgExt_AddIndicesAttr : AttrDef<IREELinalgExt_Dialect, "AddIndices", [
  DeclareAttrInterfaceMethods<IREELinalgExt_IndexTransformationInterfaceAttr, [
    "getNumResultIndices",
    "getNumInputIndices",
    "getNumDynamicIndices",
    "transformIndices",
  ]>
]> {
  let mnemonic = "add_indices";
  let cppNamespace = "::mlir::iree_compiler::IREE::LinalgExt";

  let description = [{
    Attribute describing addition of index values to a set of indices.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "values">:$values
  );

  let assemblyFormat = "custom<DynamicI64ArrayAttr>($values)";
}

#endif // IREE_COMPILER_DIALECT_LINALGEXT_LINALGEXTATTRS
