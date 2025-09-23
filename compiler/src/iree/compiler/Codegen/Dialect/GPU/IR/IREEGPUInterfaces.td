// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES

include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.td"
include "mlir/IR/EnumAttr.td"

def IREEGPU_MmaInterfaceAttr
    : AttrInterface<"MmaInterfaceAttr", [IREECodegen_InnerTileDescAttrInterface]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = [{
    Interface used to query information needed to generate code targeting a
    specific Matrix Multiply-Accumulate (MMA) machine intrinsic instruction.
    Layout information can be explicit or opaque, however all such attributes
    must specify the static shape of the operation and the required element
    types.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the subgroup size needed to execute this kind of multi-MMA.
      }],
      /*retType=*/"int64_t",
      /*methodName=*/"getSubgroupSize",
      /*args=*/(ins)
    >,
  ];

  // Methods included to preserve interfaces from beforet he InnerTileDescAttrInterface
  // refactoring.
  let extraSharedClassDeclaration = [{
    /// Returns the shape of the MMA operation:
    /// ```
    ///   C += A * B
    /// ```
    /// Where, up to some transpositions, `A` has shape `<M, K>`, `B` has shape
    /// `<K, N>`, and `C` has shape `<M, N>`.
    ::std::tuple<int64_t, int64_t, int64_t>
    getMNKShape() const {
      ::llvm::SmallVector<::mlir::VectorType> preThreadTypes;
      $_attr.getUndistributedTileTypes(preThreadTypes);
      ::llvm::ArrayRef<int64_t> accShape = preThreadTypes[2].getShape();
      ::llvm::ArrayRef<int64_t> lhsShape = preThreadTypes[0].getShape();
      return {accShape[0], accShape[1], lhsShape[1]};
    }

    /// Returns the element types of the operands to the MMA operation:
    /// ```
    ///   C += A * B
    /// ```
    ::std::tuple<::mlir::Type, ::mlir::Type, ::mlir::Type>
    getABCElementTypes() const {
      ::llvm::SmallVector<::mlir::Type> elementTypes;
      $_attr.getElementTypes(elementTypes);
      return {elementTypes[0], elementTypes[1], elementTypes[2]};
    }

    /// Returns the vector types of the operands to the MMA operation:
    /// ```
    /// C += A * B
    /// ```
    /// Each thread participating in the MMA operation is responsible for
    /// storing such shaped vectors in the registers.
    ::std::tuple<::mlir::VectorType, ::mlir::VectorType, ::mlir::VectorType>
    getABCVectorTypes() const {
      ::llvm::SmallVector<::mlir::VectorType> threadTypes;
      $_attr.getDistributedTileTypes(threadTypes);
      return {threadTypes[0], threadTypes[1], threadTypes[2]};
    }
  }];
}

// def IREEGPU_ScaledMmaInterfaceAttr
//     : AttrInterface<"ScaledMmaInterfaceAttr", [IREECodegen_InnerTileDescAttrInterface]> {
//   let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

//   let description = [{
//     Interface used to query information needed to generate code targeting a
//     specific Scaled Matrix Multiply-Accumulate (ScaledMMA) machine intrinsic
//     instruction. Layout information can be explicit or opaque, however all
//     such attributes must specify the static shape of the operation and the
//     required element types.
//   }];

//   // Methods included to preserve interfaces from beforet he InnerTileDescAttrInterface
//   // refactoring.
//   let extraClassDeclaration = [{
//     // Return all supported element types for inputs
//     static SmallVector<Type> getSupportedInputTypes(MLIRContext *ctx);
//     static SmallVector<Type> getSupportedOutputTypes(MLIRContext *ctx);

//     /// For scaled MFMA intrinsics, the scales are packed into a 32 bit register.
//     /// Since scales are 8 bits, the intrinsic takes a vector of 4 scales, along
//     /// with a selector to indicate which scale is used.
//     int64_t getScalesVectorSize() const {
//       return 4;
//     }

//     // Return the number of elements per shared scale.
//     int64_t getBlockSize() const;

//     // Return preferred subgroup size
//     int64_t getSubgroupSize() const;

//     /// Returns the shape of the scaled MMA operation:
//     /// ```
//     ///   C += A * B
//     /// ```
//     /// Where, up to some transpositions, `A` has shape `<M, K, KB>`, `B`
//     /// has shape `<K, KB, N>`, and `C` has shape `<M, N>`.
//     ::std::tuple<int64_t, int64_t, int64_t, int64_t>
//     getScaledMNKShape() const {
//       ::llvm::SmallVector<::mlir::VectorType> preThreadTypes;
//       getUndistributedTileTypes(preThreadTypes);
//       ::llvm::ArrayRef<int64_t> accShape = preThreadTypes[4].getShape();
//       ::llvm::ArrayRef<int64_t> lhsShape = preThreadTypes[0].getShape();
//       return {accShape[0], accShape[1], lhsShape[1], lhsShape[2]};
//     }
//   }];
// }

def IREEGPU_AnyMmaAttr : Attr<Or<[
  CPred<"isa<IREE::GPU::MmaInterfaceAttr>($_self)">,
]>, "buffer-like constant attribute values"> {
  let storageType = [{ IREE::GPU::MmaInterfaceAttr }];
  let returnType = [{ IREE::GPU::MmaInterfaceAttr }];
  let convertFromStorage = "$_self";
}

def IREEGPU_PromotionAttr
    : AttrInterface<"PromotionAttr"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = [{
    Interface used to control promotion of operands.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the promoted operand. The operation owning the operand cannot be
        modified by this method.
      }],
      /*retType=*/"::mlir::Value",
      /*methodName=*/"promoteOperand",
      /*args=*/(ins
        "::mlir::OpBuilder&":$builder,
        "::mlir::OpOperand&":$operand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::iree_compiler::IREE::GPU::defaultPromotionImpl(builder, operand, $_attr);
      }]
    >,
  ];
}

def IREEGPU_DataTiledMMAInterfaceAttr
    : AttrInterface<"DataTiledMMAInterfaceAttr", []> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = [{
    DataTiledMMAInterfaceAttr
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the TileSwizzle for the data-tiled MMA attribute.
      }],
      /*retType=*/"::mlir::iree_compiler::IREE::Codegen::TileSwizzle",
      /*methodName=*/"getTileSwizzle",
      /*args=*/(ins "unsigned":$operand_index)
    >,
    InterfaceMethod<
      /*desc=*/[{
        getTileMNK
      }],
      /*retType=*/"::mlir::iree_compiler::IREE::Codegen::TileMxNxK",
      /*methodName=*/"getTileMNK",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        getDataTiledElementTypes
      }],
      /*retType=*/"void",
      /*methodName=*/"getDataTiledElementTypes",
      /*args=*/(ins "SmallVectorImpl<Type>&":$result)
    >,
    InterfaceMethod<
      /*desc=*/[{
        getSubgroupSize
      }],
      /*retType=*/"int64_t",
      /*methodName=*/"getSubgroupSize",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        getFlatWorkgroupSize
      }],
      /*retType=*/"int64_t",
      /*methodName=*/"getFlatWorkgroupSize",
      /*args=*/(ins)
    >,
  ];

  let extraClassDeclaration = [{
    //========================================================================//
    // Declare interface methods for InnerTileDescAttrInterface with shared
    // implementation among DataTiledMMAInterfaceAttr.
    //========================================================================//

    void getUndistributedTileTypes(SmallVectorImpl<VectorType>& result);

    void getDistributedTileTypes(SmallVectorImpl<VectorType>& result);

    std::optional<::mlir::SmallVector<int64_t, 2>>
    getUndistributedTileDimExpansion(int64_t operandIndex, int64_t logicalDim);

    LogicalResult populateOperandOffsetsSizesStrides(
        OpBuilder &builder, Location loc, uint32_t operandIndex, Value laneId,
        ArrayRef<int64_t> permutation, SmallVectorImpl<OpFoldResult> &offsets,
        SmallVectorImpl<OpFoldResult> &sizes,
        SmallVectorImpl<OpFoldResult> &strides);

    Attribute getDistributionMappingKind();

    OpFoldResult getDistributionWorkerCount(
        OpBuilder &builder, Location loc, Operation *opToDistribute);

    // LogicalResult buildUnderlyingOperations(
    //     OpBuilder &builder, Location loc, ValueRange inputs, ValueRange outputs,
    //     llvm::SmallVectorImpl<Value>& results);
  }];
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES
