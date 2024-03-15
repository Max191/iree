// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/gguf/gguf_parser.h"
#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/io/formats/safetensors/safetensors_parser.h"

#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/ArchiveUtils.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/FileUtilities.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

#define GEN_PASS_DEF_IMPORTPARAMETERSPASS
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h.inc"

namespace {

using FileHandle =
    std::unique_ptr<iree_io_file_handle_t, void (*)(iree_io_file_handle_t *)>;
using ParameterIndex = std::unique_ptr<iree_io_parameter_index_t,
                                       void (*)(iree_io_parameter_index_t *)>;

static FailureOr<FileHandle> openArchiveFile(ModuleOp moduleOp,
                                             StringRef archivePath) {
  iree_allocator_t hostAllocator = iree_allocator_system();

  // Open the archive (hopefully mapped).
  auto fileOrErr = llvm::MemoryBuffer::getFile(
      archivePath, /*IsText=*/false, /*RequiresNullTerminator=*/false,
      /*IsVolatile=*/false, /*Alignment=*/std::nullopt);
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << "cannot open archive input file '" + archivePath +
                        "': " + error.message();
    return failure();
  }
  auto file = std::move(fileOrErr.get());

  // A callback issued when a file is released to destroy the file.
  iree_io_file_handle_release_callback_t fileReleaseCallback;
  fileReleaseCallback.fn =
      +[](void *user_data, iree_io_file_handle_primitive_t handle_primitive) {
        delete reinterpret_cast<llvm::MemoryBuffer *>(user_data);
      };
  fileReleaseCallback.user_data = file.get();

  // Wrap the archive in a file handle.
  iree_io_file_handle_t *fileHandle = nullptr;
  if (failed(handleRuntimeError(
          moduleOp,
          iree_io_file_handle_wrap_host_allocation(
              IREE_IO_FILE_ACCESS_READ,
              iree_make_byte_span(const_cast<char *>(file->getBufferStart()),
                                  file->getBufferSize()),
              fileReleaseCallback, hostAllocator, &fileHandle),
          "unable to wrap archive memory buffer"))) {
    return failure();
  }
  file.release(); // now owned by the fileHandle

  return FileHandle(fileHandle, iree_io_file_handle_release);
}

static FailureOr<ParameterIndex> loadParameterIndex(ModuleOp moduleOp,
                                                    StringRef archivePath) {
  iree_allocator_t hostAllocator = iree_allocator_system();

  // Open the archive file (hopefully mapping it).
  auto fileHandle = openArchiveFile(moduleOp, archivePath);
  if (failed(fileHandle))
    return failure();

  // Allocate the parameter index we'll populate with the archive file.
  iree_io_parameter_index_t *parameterIndexPtr = nullptr;
  if (failed(handleRuntimeError(
          moduleOp,
          iree_io_parameter_index_create(hostAllocator, &parameterIndexPtr),
          "unable to allocate empty parameter index"))) {
    return failure();
  }
  auto parameterIndex =
      ParameterIndex(parameterIndexPtr, iree_io_parameter_index_release);

  // Parse the archive as a particular format.
  // TODO(benvanik): centralize this type selection logic in iree/io/.
  StringRef format = llvm::sys::path::extension(archivePath);
  if (format == ".gguf") {
    if (failed(handleRuntimeError(
            moduleOp,
            iree_io_parse_gguf_index(fileHandle->get(), parameterIndex.get()),
            "parsing gguf file"))) {
      return failure();
    }
  } else if (format == ".irpa") {
    if (failed(handleRuntimeError(
            moduleOp,
            iree_io_parse_irpa_index(fileHandle->get(), parameterIndex.get()),
            "parsing irpa file"))) {
      return failure();
    }
  } else if (format == ".safetensors") {
    if (failed(handleRuntimeError(moduleOp,
                                  iree_io_parse_safetensors_index(
                                      fileHandle->get(), parameterIndex.get()),
                                  "parsing safetensors file"))) {
      return failure();
    }
  } else {
    llvm::errs() << "unsupported archive file format: " << archivePath << "\n";
    return failure();
  }

  return parameterIndex;
}

// Today only shaped types of elements where we know we can directly access the
// data as stored in teh file.
static bool isTypeSupported(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType)
    return false;
  auto elementType = shapedType.getElementType();
  // NOTE: packed types not yet supported.
  if (!elementType.isIntOrFloat())
    return false;
  const unsigned logicalBitWidth = elementType.getIntOrFloatBitWidth();
  switch (logicalBitWidth) {
  case 8:
  case 16:
  case 32:
  case 64:
    return true;
  default:
    return false;
  }
}

static LogicalResult
importParameterFromSplat(StringRef fullName,
                         IREE::Util::GlobalOpInterface globalOp,
                         const iree_io_parameter_index_entry_t *entry) {
  entry->storage.splat.pattern;
  entry->storage.splat.pattern_length;

  // Map the splat pattern into an attribute.
  auto shapedType = cast<ShapedType>(globalOp.getGlobalType());
  auto elementType = shapedType.getElementType();
  Attribute valueAttr;
  if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    // DO NOT SUBMIT
  } else if (auto floatType = dyn_cast<FloatType>(elementType)) {
    // DO NOT SUBMIT
  } else if (auto complexType = dyn_cast<ComplexType>(elementType)) {
    // DO NOT SUBMIT
  }
  if (!valueAttr) {
    llvm::errs() << "unsupported splat type: " << elementType << "\n";
    return failure();
  }

  // Update the global to use the splat value.
  globalOp.setGlobalInitialValue(SplatElementsAttr::get(
      cast<ShapedType>(globalOp.getGlobalType()), valueAttr));

  return success();
}

// TODO(benvanik): replace with resources, maybe? there's no FileAsmResourceBlob
// yet but we could use that to point back to the file on disk. For now we just
// import as a raw attr to ensure that imported parameters behave exactly as
// constants would everywhere and can be serialized/deserialized across
// reproducers/etc.
static LogicalResult
importParameterFromFile(StringRef fullName,
                        IREE::Util::GlobalOpInterface globalOp,
                        const iree_io_parameter_index_entry_t *entry) {
  // We currently only support mapped files, but could instead handle file path
  // references and point resource blobs directly at them.
  iree_io_file_handle_primitive_t filePrimitive =
      iree_io_file_handle_primitive(entry->storage.file.handle);
  if (filePrimitive.type != IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    llvm::errs() << "only host allocation file primitives are supported\n";
    return failure();
  }
  const uint8_t *fileData = filePrimitive.value.host_allocation.data;

  // Copy the data from the parameter file into an attribute and change the
  // global to use it as its initial value.
  globalOp.setGlobalInitialValue(DenseElementsAttr::getFromRawBuffer(
      cast<ShapedType>(globalOp.getGlobalType()),
      ArrayRef<char>(
          reinterpret_cast<const char *>(fileData + entry->storage.file.offset),
          entry->length)));

  return success();
}

static LogicalResult
tryImportParameter(StringRef fullName, IREE::Util::GlobalOpInterface globalOp,
                   IREE::Stream::NamedParameterAttr parameterAttr,
                   iree_io_parameter_index_t *parameterIndex) {
  // Lookup the parameter in the index - it may not be present.
  auto key = parameterAttr.getKey().getValue();
  const iree_io_parameter_index_entry_t *entry = nullptr;
  iree_status_t lookupStatus = iree_io_parameter_index_lookup(
      parameterIndex, iree_make_string_view(key.data(), key.size()), &entry);
  if (!iree_status_is_ok(lookupStatus)) {
    iree_status_ignore(lookupStatus);
    return success(); // not found is ok
  }

  switch (entry->type) {
  case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT:
    return importParameterFromSplat(fullName, globalOp, entry);
  case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE:
    return importParameterFromFile(fullName, globalOp, entry);
  default:
    // Unsupported type.
    llvm::errs() << "found parameter but type is not supported: " << key
                 << "\n";
    return failure();
  }
}

struct ImportParametersPass
    : public IREE::IO::Parameters::impl::ImportParametersPassBase<
          ImportParametersPass> {
  using IREE::IO::Parameters::impl::ImportParametersPassBase<
      ImportParametersPass>::ImportParametersPassBase;

  void runOnOperation() override {
    // Nothing to do if no path specified.
    if (archivePath.empty()) {
      return;
    }

    MLIRContext *context = &getContext();
    ModuleOp moduleOp = getOperation();

    // Open the archive file (hopefully mapping it) and parse the index.
    auto parameterIndex = loadParameterIndex(moduleOp, archivePath);
    if (failed(parameterIndex))
      return signalPassFailure();

    // Decide whether to import a particular parameter.
    DenseSet<StringRef> importKeys;
    for (auto &key : parameterKeys)
      importKeys.insert(key);
    auto shouldImportParameter =
        [&](IREE::Stream::NamedParameterAttr parameterAttr) -> bool {
      // If a scope is defined filter to just that scope.
      if (!parameterScope.empty() &&
          parameterAttr.getScope() != parameterScope) {
        return false; // scope mismatch
      }
      // Always try to import explicitly named parameters.
      if (importKeys.contains(parameterAttr.getKey().getValue())) {
        return true; // key match
      }
      // If a maximum size is specified use that to limit what we import
      // (users may want to bring in small parameters but leave the big ones
      // out).
      if (maximumSize && parameterAttr.getStorageSize() <= maximumSize) {
        return true; // <= max size
      }
      // Default to not importing.
      return false;
    };

    // Find all parameters and try to import them.
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
      if (!isTypeSupported(globalOp.getGlobalType()))
        continue;
      if (auto parameterAttr =
              dyn_cast_if_present<IREE::Stream::NamedParameterAttr>(
                  globalOp.getGlobalInitialValue())) {
        if (shouldImportParameter(parameterAttr)) {
          std::string fullName =
              (StringRef("__import_") + parameterAttr.getScope().getValue() +
               "_" + parameterAttr.getKey().getValue())
                  .str();
          if (failed(tryImportParameter(fullName, globalOp, parameterAttr,
                                        parameterIndex->get()))) {
            return signalPassFailure();
          }
        }
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::IO::Parameters
