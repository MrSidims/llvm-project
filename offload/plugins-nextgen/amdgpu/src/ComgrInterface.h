//===-- ComgrInterface.h - AMD comgr dynamic loading interface ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a thin wrapper around AMD's comgr (Code Object Manager)
// library for compiling SPIR-V to native AMDGPU code objects at runtime.
// The library is loaded dynamically via dlopen to avoid a hard build dependency.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_AMDGPU_COMGRINTERFACE_H
#define OPENMP_LIBOMPTARGET_PLUGINS_AMDGPU_COMGRINTERFACE_H

#include "Shared/Debug.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

/// Comgr type and enum declarations matching amd_comgr.h.
/// We declare them here to avoid requiring the comgr header at build time.
typedef enum {
  AMD_COMGR_STATUS_SUCCESS = 0x0,
  AMD_COMGR_STATUS_ERROR = 0x1,
  AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT = 0x2,
  AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES = 0x3,
} amd_comgr_status_t;

typedef enum {
  AMD_COMGR_DATA_KIND_UNDEF = 0x0,
  AMD_COMGR_DATA_KIND_SOURCE = 0x1,
  AMD_COMGR_DATA_KIND_INCLUDE = 0x2,
  AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER = 0x3,
  AMD_COMGR_DATA_KIND_DIAGNOSTIC = 0x4,
  AMD_COMGR_DATA_KIND_LOG = 0x5,
  AMD_COMGR_DATA_KIND_BC = 0x6,
  AMD_COMGR_DATA_KIND_RELOCATABLE = 0x7,
  AMD_COMGR_DATA_KIND_EXECUTABLE = 0x8,
  AMD_COMGR_DATA_KIND_BYTES = 0x9,
  AMD_COMGR_DATA_KIND_FATBIN = 0xA,
  AMD_COMGR_DATA_KIND_AR = 0xB,
  AMD_COMGR_DATA_KIND_BC_BUNDLE = 0xC,
  AMD_COMGR_DATA_KIND_AR_BUNDLE = 0xD,
  AMD_COMGR_DATA_KIND_SPIRV = 0xE,
} amd_comgr_data_kind_t;

typedef enum {
  AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR = 0x0,
  AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS = 0x1,
  AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC = 0x2,
  AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES = 0x3,
  AMD_COMGR_ACTION_LINK_BC_TO_BC = 0x4,
  AMD_COMGR_ACTION_OPTIMIZE_BC_TO_BC = 0x5,
  AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE = 0x6,
  AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY = 0x7,
  AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE = 0x8,
  AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 0x9,
  AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE = 0xA,
  AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE = 0xB,
  AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE = 0xC,
  AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE = 0xD,
  AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN = 0xE,
  AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC = 0xF,
  AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC = 0x10,
} amd_comgr_action_kind_t;

typedef enum {
  AMD_COMGR_LANGUAGE_NONE = 0x0,
  AMD_COMGR_LANGUAGE_OPENCL_1_2 = 0x1,
  AMD_COMGR_LANGUAGE_OPENCL_2_0 = 0x2,
  AMD_COMGR_LANGUAGE_HC = 0x3,
  AMD_COMGR_LANGUAGE_HIP = 0x4,
} amd_comgr_language_t;

typedef struct {
  uint64_t handle;
} amd_comgr_data_t;

typedef struct {
  uint64_t handle;
} amd_comgr_data_set_t;

typedef struct {
  uint64_t handle;
} amd_comgr_action_info_t;

/// Comgr function pointer types.
using comgr_create_data_set_t = amd_comgr_status_t (*)(amd_comgr_data_set_t *);
using comgr_destroy_data_set_t = amd_comgr_status_t (*)(amd_comgr_data_set_t);
using comgr_create_data_t =
    amd_comgr_status_t (*)(amd_comgr_data_kind_t, amd_comgr_data_t *);
using comgr_release_data_t = amd_comgr_status_t (*)(amd_comgr_data_t);
using comgr_set_data_t =
    amd_comgr_status_t (*)(amd_comgr_data_t, size_t, const char *);
using comgr_set_data_name_t =
    amd_comgr_status_t (*)(amd_comgr_data_t, const char *);
using comgr_data_set_add_t =
    amd_comgr_status_t (*)(amd_comgr_data_set_t, amd_comgr_data_t);
using comgr_create_action_info_t =
    amd_comgr_status_t (*)(amd_comgr_action_info_t *);
using comgr_destroy_action_info_t =
    amd_comgr_status_t (*)(amd_comgr_action_info_t);
using comgr_action_info_set_isa_name_t =
    amd_comgr_status_t (*)(amd_comgr_action_info_t, const char *);
using comgr_action_info_set_logging_t =
    amd_comgr_status_t (*)(amd_comgr_action_info_t, bool);
using comgr_do_action_t =
    amd_comgr_status_t (*)(amd_comgr_action_kind_t, amd_comgr_action_info_t,
                           amd_comgr_data_set_t, amd_comgr_data_set_t);
using comgr_action_data_count_t =
    amd_comgr_status_t (*)(amd_comgr_data_set_t, amd_comgr_data_kind_t,
                           size_t *);
using comgr_action_data_get_data_t =
    amd_comgr_status_t (*)(amd_comgr_data_set_t, amd_comgr_data_kind_t,
                           size_t, amd_comgr_data_t *);
using comgr_get_data_t =
    amd_comgr_status_t (*)(amd_comgr_data_t, size_t *, char *);
using comgr_status_string_t =
    amd_comgr_status_t (*)(amd_comgr_status_t, const char **);

/// Interface for dynamically loading and using AMD's comgr library
/// to compile SPIR-V to native AMDGPU code objects.
class ComgrInterface {
  /// Function pointers for comgr API.
  comgr_create_data_set_t CreateDataSet = nullptr;
  comgr_destroy_data_set_t DestroyDataSet = nullptr;
  comgr_create_data_t CreateData = nullptr;
  comgr_release_data_t ReleaseData = nullptr;
  comgr_set_data_t SetData = nullptr;
  comgr_set_data_name_t SetDataName = nullptr;
  comgr_data_set_add_t DataSetAdd = nullptr;
  comgr_create_action_info_t CreateActionInfo = nullptr;
  comgr_destroy_action_info_t DestroyActionInfo = nullptr;
  comgr_action_info_set_isa_name_t ActionInfoSetIsaName = nullptr;
  comgr_action_info_set_logging_t ActionInfoSetLogging = nullptr;
  comgr_do_action_t DoAction = nullptr;
  comgr_action_data_count_t ActionDataCount = nullptr;
  comgr_action_data_get_data_t ActionDataGetData = nullptr;
  comgr_get_data_t GetData = nullptr;
  comgr_status_string_t StatusString = nullptr;

  bool Initialized = false;
  bool Available = false;
  std::once_flag InitFlag;

  void initOnce() {
    std::string ErrMsg;
    auto Lib = llvm::sys::DynamicLibrary::getPermanentLibrary(
        "libamd_comgr.so", &ErrMsg);
    if (!Lib.isValid()) {
      REPORT("Could not load comgr library: %s\n", ErrMsg.c_str());
      return;
    }

#define LOAD_COMGR_SYM(Name, Field)                                            \
  Field = reinterpret_cast<decltype(Field)>(                                   \
      Lib.getAddressOfSymbol("amd_comgr_" #Name));                            \
  if (!Field) {                                                                \
    REPORT("Could not find symbol 'amd_comgr_" #Name "' in comgr library\n"); \
    return;                                                                    \
  }

    LOAD_COMGR_SYM(create_data_set, CreateDataSet)
    LOAD_COMGR_SYM(destroy_data_set, DestroyDataSet)
    LOAD_COMGR_SYM(create_data, CreateData)
    LOAD_COMGR_SYM(release_data, ReleaseData)
    LOAD_COMGR_SYM(set_data, SetData)
    LOAD_COMGR_SYM(set_data_name, SetDataName)
    LOAD_COMGR_SYM(data_set_add, DataSetAdd)
    LOAD_COMGR_SYM(create_action_info, CreateActionInfo)
    LOAD_COMGR_SYM(destroy_action_info, DestroyActionInfo)
    LOAD_COMGR_SYM(action_info_set_isa_name, ActionInfoSetIsaName)
    LOAD_COMGR_SYM(action_info_set_logging, ActionInfoSetLogging)
    LOAD_COMGR_SYM(do_action, DoAction)
    LOAD_COMGR_SYM(action_data_count, ActionDataCount)
    LOAD_COMGR_SYM(action_data_get_data, ActionDataGetData)
    LOAD_COMGR_SYM(get_data, GetData)
    LOAD_COMGR_SYM(status_string, StatusString)

#undef LOAD_COMGR_SYM

    Available = true;
  }

  /// Get a human-readable error string for a comgr status code.
  std::string getStatusString(amd_comgr_status_t Status) const {
    const char *Str = nullptr;
    if (StatusString && StatusString(Status, &Str) == AMD_COMGR_STATUS_SUCCESS)
      return std::string(Str);
    return "unknown comgr error (status " + std::to_string((int)Status) + ")";
  }

  /// RAII wrapper for comgr data sets.
  struct DataSetRAII {
    amd_comgr_data_set_t Set;
    const ComgrInterface &CI;
    bool Valid = false;

    DataSetRAII(const ComgrInterface &CI) : CI(CI) {
      Valid = (CI.CreateDataSet(&Set) == AMD_COMGR_STATUS_SUCCESS);
    }
    ~DataSetRAII() {
      if (Valid)
        CI.DestroyDataSet(Set);
    }
    operator amd_comgr_data_set_t() const { return Set; }
  };

  /// RAII wrapper for comgr action info.
  struct ActionInfoRAII {
    amd_comgr_action_info_t Info;
    const ComgrInterface &CI;
    bool Valid = false;

    ActionInfoRAII(const ComgrInterface &CI) : CI(CI) {
      Valid = (CI.CreateActionInfo(&Info) == AMD_COMGR_STATUS_SUCCESS);
    }
    ~ActionInfoRAII() {
      if (Valid)
        CI.DestroyActionInfo(Info);
    }
    operator amd_comgr_action_info_t() const { return Info; }
  };

  /// RAII wrapper for comgr data.
  struct DataRAII {
    amd_comgr_data_t Data;
    const ComgrInterface &CI;
    bool Valid = false;

    DataRAII(const ComgrInterface &CI, amd_comgr_data_kind_t Kind) : CI(CI) {
      Valid = (CI.CreateData(Kind, &Data) == AMD_COMGR_STATUS_SUCCESS);
    }
    ~DataRAII() {
      if (Valid)
        CI.ReleaseData(Data);
    }
    operator amd_comgr_data_t() const { return Data; }
  };

  /// Extract binary data from a data set matching the given kind.
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  extractFromDataSet(amd_comgr_data_set_t DataSet,
                     amd_comgr_data_kind_t Kind) const {
    size_t Count = 0;
    amd_comgr_status_t Status = ActionDataCount(DataSet, Kind, &Count);
    if (Status != AMD_COMGR_STATUS_SUCCESS)
      return llvm::createStringError("comgr: failed to count output data: %s",
                                     getStatusString(Status).c_str());

    if (Count == 0)
      return llvm::createStringError(
          "comgr: no output data of requested kind produced");

    amd_comgr_data_t OutData;
    Status = ActionDataGetData(DataSet, Kind, 0, &OutData);
    if (Status != AMD_COMGR_STATUS_SUCCESS)
      return llvm::createStringError("comgr: failed to get output data: %s",
                                     getStatusString(Status).c_str());

    // Get the size first.
    size_t Size = 0;
    Status = GetData(OutData, &Size, nullptr);
    if (Status != AMD_COMGR_STATUS_SUCCESS) {
      ReleaseData(OutData);
      return llvm::createStringError(
          "comgr: failed to get output data size: %s",
          getStatusString(Status).c_str());
    }

    // Allocate and read the data.
    auto Buf = llvm::WritableMemoryBuffer::getNewMemBuffer(Size);
    Status = GetData(OutData, &Size, Buf->getBufferStart());
    ReleaseData(OutData);

    if (Status != AMD_COMGR_STATUS_SUCCESS)
      return llvm::createStringError("comgr: failed to read output data: %s",
                                     getStatusString(Status).c_str());

    return std::unique_ptr<llvm::MemoryBuffer>(std::move(Buf));
  }

public:
  /// Ensure the library is loaded. Thread-safe, loads at most once.
  void init() { std::call_once(InitFlag, [this]() { initOnce(); }); }

  /// Return true if comgr library was loaded successfully.
  bool isAvailable() const { return Available; }

  /// Compile a SPIR-V binary to an AMDGPU code object for the given ISA.
  /// \param SPIRVImage  The SPIR-V binary data.
  /// \param ISAName     The target ISA (e.g., "amdgcn-amd-amdhsa--gfx90a").
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  compileSPIRV(llvm::StringRef SPIRVImage, llvm::StringRef ISAName) const {
    if (!Available)
      return llvm::createStringError(
          "SPIR-V compilation requires AMD comgr library "
          "(libamd_comgr.so). Install ROCm to enable SPIR-V support.");

    // Create action info with ISA name.
    ActionInfoRAII ActionInfo(*this);
    if (!ActionInfo.Valid)
      return llvm::createStringError(
          "comgr: failed to create action info");

    amd_comgr_status_t Status =
        ActionInfoSetIsaName(ActionInfo, ISAName.data());
    if (Status != AMD_COMGR_STATUS_SUCCESS)
      return llvm::createStringError("comgr: failed to set ISA name '%s': %s",
                                     ISAName.data(),
                                     getStatusString(Status).c_str());

    ActionInfoSetLogging(ActionInfo, true);

    // Step 1: Create input data set with SPIR-V data.
    DataSetRAII InputSet(*this);
    if (!InputSet.Valid)
      return llvm::createStringError(
          "comgr: failed to create input data set");

    DataRAII SPIRVData(*this, AMD_COMGR_DATA_KIND_SPIRV);
    if (!SPIRVData.Valid)
      return llvm::createStringError(
          "comgr: failed to create SPIR-V data object");

    Status = SetData(SPIRVData, SPIRVImage.size(), SPIRVImage.data());
    if (Status != AMD_COMGR_STATUS_SUCCESS)
      return llvm::createStringError("comgr: failed to set SPIR-V data: %s",
                                     getStatusString(Status).c_str());

    Status = SetDataName(SPIRVData, "input.spv");
    if (Status != AMD_COMGR_STATUS_SUCCESS)
      return llvm::createStringError(
          "comgr: failed to set SPIR-V data name: %s",
          getStatusString(Status).c_str());

    Status = DataSetAdd(InputSet, SPIRVData);
    if (Status != AMD_COMGR_STATUS_SUCCESS)
      return llvm::createStringError(
          "comgr: failed to add SPIR-V data to data set: %s",
          getStatusString(Status).c_str());

    // Step 2: SPIR-V -> LLVM Bitcode.
    DataSetRAII BCSet(*this);
    if (!BCSet.Valid)
      return llvm::createStringError(
          "comgr: failed to create bitcode output data set");

    Status = DoAction(AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC, ActionInfo,
                      InputSet, BCSet);
    if (Status != AMD_COMGR_STATUS_SUCCESS)
      return llvm::createStringError(
          "comgr: failed to translate SPIR-V to bitcode: %s",
          getStatusString(Status).c_str());

    // Step 3: Bitcode -> Relocatable object.
    DataSetRAII RelocSet(*this);
    if (!RelocSet.Valid)
      return llvm::createStringError(
          "comgr: failed to create relocatable output data set");

    Status = DoAction(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, ActionInfo,
                      BCSet, RelocSet);
    if (Status != AMD_COMGR_STATUS_SUCCESS)
      return llvm::createStringError(
          "comgr: failed to generate relocatable from bitcode: %s",
          getStatusString(Status).c_str());

    // Step 4: Relocatable -> Executable code object.
    DataSetRAII ExecSet(*this);
    if (!ExecSet.Valid)
      return llvm::createStringError(
          "comgr: failed to create executable output data set");

    Status = DoAction(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                      ActionInfo, RelocSet, ExecSet);
    if (Status != AMD_COMGR_STATUS_SUCCESS)
      return llvm::createStringError(
          "comgr: failed to link relocatable to executable: %s",
          getStatusString(Status).c_str());

    // Extract the executable code object.
    return extractFromDataSet(ExecSet, AMD_COMGR_DATA_KIND_EXECUTABLE);
  }
};

#endif // OPENMP_LIBOMPTARGET_PLUGINS_AMDGPU_COMGRINTERFACE_H
