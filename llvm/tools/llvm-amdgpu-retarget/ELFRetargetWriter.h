//===-- ELFRetargetWriter.h - ELF rewriting for retargeting -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for rewriting AMDGPU ELF code objects with
// new .text section content and updated ELF flags.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_AMDGPU_RETARGET_ELFRETARGETWRITER_H
#define LLVM_TOOLS_LLVM_AMDGPU_RETARGET_ELFRETARGETWRITER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <vector>

namespace llvm {

class MemoryBuffer;

namespace amdgpu {

/// Information about a section to be modified during retargeting.
struct SectionPatch {
  std::string Name;
  std::vector<uint8_t> NewContent;
  uint64_t OriginalOffset;
  uint64_t OriginalSize;
};

/// Get the ELF machine flag for an AMDGPU target.
/// Returns 0 if the target is unknown.
unsigned getAMDGPUElfMach(StringRef CPU);

/// Rewrite an AMDGPU ELF code object with new .text content and updated flags.
///
/// This function:
/// 1. Parses the input ELF
/// 2. Replaces .text section content with new bytes
/// 3. Updates the e_flags to reflect the target architecture
/// 4. Updates the .note.AMD section if present
/// 5. Optionally updates kernel descriptor VGPR counts
/// 6. Writes the modified ELF to the output
///
/// \param InputBuffer The original ELF file contents.
/// \param NewTextSection The new .text section content.
/// \param SourceCPU The source GPU architecture (e.g., "gfx942").
/// \param TargetCPU The target GPU architecture (e.g., "gfx90a").
/// \param OutputPath Path to write the output file.
/// \param Verbose Enable verbose output.
/// \param ExtraVGPRs Additional VGPRs to add to kernel descriptor (for emulation).
/// \returns Error on failure.
Error rewriteELFWithNewText(const MemoryBuffer &InputBuffer,
                            ArrayRef<char> NewTextSection,
                            StringRef SourceCPU,
                            StringRef TargetCPU,
                            StringRef OutputPath,
                            bool Verbose,
                            unsigned ExtraVGPRs = 0);

/// Update kernel descriptor VGPR counts in an ELF buffer.
/// This is used when instruction emulation requires additional VGPRs.
///
/// The kernel descriptor is located at the start of each kernel's code
/// in the .text section. The VGPR count is stored in RSRC1.
///
/// \param ELFData Mutable ELF data buffer.
/// \param TextOffset Offset of .text section in ELF.
/// \param TextSize Size of .text section.
/// \param ExtraVGPRs Number of additional VGPRs needed.
/// \param Verbose Enable verbose output.
void updateKernelDescriptorVGPRs(std::vector<uint8_t> &ELFData,
                                 uint64_t TextOffset,
                                 uint64_t TextSize,
                                 unsigned ExtraVGPRs,
                                 bool Verbose);

} // namespace amdgpu
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_AMDGPU_RETARGET_ELFRETARGETWRITER_H
