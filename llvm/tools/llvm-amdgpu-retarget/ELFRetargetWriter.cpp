//===-- ELFRetargetWriter.cpp - ELF rewriting for retargeting -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for rewriting AMDGPU ELF code objects.
//
//===----------------------------------------------------------------------===//

#include "ELFRetargetWriter.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::amdgpu;
using namespace llvm::msgpack;

unsigned llvm::amdgpu::getAMDGPUElfMach(StringRef CPU) {
  // Map CPU name to ELF machine flag
  return StringSwitch<unsigned>(CPU)
      .Case("gfx600", ELF::EF_AMDGPU_MACH_AMDGCN_GFX600)
      .Case("gfx601", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601)
      .Case("gfx602", ELF::EF_AMDGPU_MACH_AMDGCN_GFX602)
      .Case("gfx700", ELF::EF_AMDGPU_MACH_AMDGCN_GFX700)
      .Case("gfx701", ELF::EF_AMDGPU_MACH_AMDGCN_GFX701)
      .Case("gfx702", ELF::EF_AMDGPU_MACH_AMDGCN_GFX702)
      .Case("gfx703", ELF::EF_AMDGPU_MACH_AMDGCN_GFX703)
      .Case("gfx704", ELF::EF_AMDGPU_MACH_AMDGCN_GFX704)
      .Case("gfx705", ELF::EF_AMDGPU_MACH_AMDGCN_GFX705)
      .Case("gfx801", ELF::EF_AMDGPU_MACH_AMDGCN_GFX801)
      .Case("gfx802", ELF::EF_AMDGPU_MACH_AMDGCN_GFX802)
      .Case("gfx803", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803)
      .Case("gfx805", ELF::EF_AMDGPU_MACH_AMDGCN_GFX805)
      .Case("gfx810", ELF::EF_AMDGPU_MACH_AMDGCN_GFX810)
      .Case("gfx900", ELF::EF_AMDGPU_MACH_AMDGCN_GFX900)
      .Case("gfx902", ELF::EF_AMDGPU_MACH_AMDGCN_GFX902)
      .Case("gfx904", ELF::EF_AMDGPU_MACH_AMDGCN_GFX904)
      .Case("gfx906", ELF::EF_AMDGPU_MACH_AMDGCN_GFX906)
      .Case("gfx908", ELF::EF_AMDGPU_MACH_AMDGCN_GFX908)
      .Case("gfx909", ELF::EF_AMDGPU_MACH_AMDGCN_GFX909)
      .Case("gfx90a", ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A)
      .Case("gfx90c", ELF::EF_AMDGPU_MACH_AMDGCN_GFX90C)
      .Case("gfx942", ELF::EF_AMDGPU_MACH_AMDGCN_GFX942)
      .Case("gfx1010", ELF::EF_AMDGPU_MACH_AMDGCN_GFX1010)
      .Case("gfx1011", ELF::EF_AMDGPU_MACH_AMDGCN_GFX1011)
      .Case("gfx1012", ELF::EF_AMDGPU_MACH_AMDGCN_GFX1012)
      .Case("gfx1013", ELF::EF_AMDGPU_MACH_AMDGCN_GFX1013)
      .Case("gfx1030", ELF::EF_AMDGPU_MACH_AMDGCN_GFX1030)
      .Case("gfx1031", ELF::EF_AMDGPU_MACH_AMDGCN_GFX1031)
      .Case("gfx1032", ELF::EF_AMDGPU_MACH_AMDGCN_GFX1032)
      .Case("gfx1033", ELF::EF_AMDGPU_MACH_AMDGCN_GFX1033)
      .Case("gfx1034", ELF::EF_AMDGPU_MACH_AMDGCN_GFX1034)
      .Case("gfx1035", ELF::EF_AMDGPU_MACH_AMDGCN_GFX1035)
      .Case("gfx1100", ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100)
      .Case("gfx1150", ELF::EF_AMDGPU_MACH_AMDGCN_GFX1150)
      .Default(0);
}

namespace {

// ELF64 header structure for direct manipulation
using Elf64_Ehdr = ELF::Elf64_Ehdr;
using Elf64_Shdr = ELF::Elf64_Shdr;
using Elf64_Phdr = ELF::Elf64_Phdr;

/// Helper class to rewrite an ELF file with modified sections.
class ELFRewriter {
public:
  ELFRewriter(const MemoryBuffer &Input, StringRef SourceCPU,
              StringRef TargetCPU, bool Verbose)
      : InputData(Input.getBufferStart()), InputSize(Input.getBufferSize()),
        SourceCPU(SourceCPU), TargetCPU(TargetCPU), Verbose(Verbose) {}

  Error rewrite(ArrayRef<char> NewTextSection, StringRef OutputPath);

private:
  Error parseELFHeader();
  Error findTextSection();
  Error buildOutput(ArrayRef<char> NewTextSection);
  Error updateNoteSections();
  Error adjustRelocations(uint64_t TextEnd, uint64_t AlignedSizeDiff);
  Error writeOutput(StringRef OutputPath);

  // Update e_flags with the new target architecture
  uint32_t computeNewFlags(uint32_t OldFlags);

  // Build the target string for the new architecture
  std::string buildTargetString() const;

  const char *InputData;
  size_t InputSize;
  StringRef SourceCPU;
  StringRef TargetCPU;
  bool Verbose;

  // Parsed ELF info
  const Elf64_Ehdr *Header = nullptr;
  const Elf64_Shdr *SectionHeaders = nullptr;
  const Elf64_Phdr *ProgramHeaders = nullptr;
  uint16_t NumSections = 0;
  uint16_t NumProgramHeaders = 0;
  const char *SectionStringTable = nullptr;

  // Text section info
  uint64_t TextSectionOffset = 0;
  uint64_t TextSectionSize = 0;
  uint16_t TextSectionIndex = 0;

  // Output buffer
  std::vector<uint8_t> OutputBuffer;
};

Error ELFRewriter::parseELFHeader() {
  if (InputSize < sizeof(Elf64_Ehdr))
    return createStringError(inconvertibleErrorCode(),
                             "File too small for ELF header");

  Header = reinterpret_cast<const Elf64_Ehdr *>(InputData);

  // Verify ELF magic
  if (Header->e_ident[ELF::EI_MAG0] != ELF::ElfMagic[0] ||
      Header->e_ident[ELF::EI_MAG1] != ELF::ElfMagic[1] ||
      Header->e_ident[ELF::EI_MAG2] != ELF::ElfMagic[2] ||
      Header->e_ident[ELF::EI_MAG3] != ELF::ElfMagic[3])
    return createStringError(inconvertibleErrorCode(), "Invalid ELF magic");

  // Verify 64-bit little-endian
  if (Header->e_ident[ELF::EI_CLASS] != ELF::ELFCLASS64)
    return createStringError(inconvertibleErrorCode(),
                             "Only 64-bit ELF is supported");

  if (Header->e_ident[ELF::EI_DATA] != ELF::ELFDATA2LSB)
    return createStringError(inconvertibleErrorCode(),
                             "Only little-endian ELF is supported");

  // Verify AMDGPU
  if (Header->e_machine != ELF::EM_AMDGPU)
    return createStringError(inconvertibleErrorCode(),
                             "Not an AMDGPU ELF file");

  NumSections = Header->e_shnum;
  NumProgramHeaders = Header->e_phnum;

  if (Header->e_shoff + NumSections * sizeof(Elf64_Shdr) > InputSize)
    return createStringError(inconvertibleErrorCode(),
                             "Section headers extend past end of file");

  SectionHeaders =
      reinterpret_cast<const Elf64_Shdr *>(InputData + Header->e_shoff);

  if (Header->e_phoff &&
      Header->e_phoff + NumProgramHeaders * sizeof(Elf64_Phdr) > InputSize)
    return createStringError(inconvertibleErrorCode(),
                             "Program headers extend past end of file");

  if (Header->e_phoff)
    ProgramHeaders =
        reinterpret_cast<const Elf64_Phdr *>(InputData + Header->e_phoff);

  // Get section string table
  if (Header->e_shstrndx >= NumSections)
    return createStringError(inconvertibleErrorCode(),
                             "Invalid section string table index");

  const Elf64_Shdr &StrTabSec = SectionHeaders[Header->e_shstrndx];
  if (StrTabSec.sh_offset + StrTabSec.sh_size > InputSize)
    return createStringError(inconvertibleErrorCode(),
                             "Section string table extends past end of file");

  SectionStringTable = InputData + StrTabSec.sh_offset;

  return Error::success();
}

Error ELFRewriter::findTextSection() {
  for (uint16_t I = 0; I < NumSections; ++I) {
    const Elf64_Shdr &Sec = SectionHeaders[I];
    if (Sec.sh_name >= SectionHeaders[Header->e_shstrndx].sh_size)
      continue;

    StringRef Name(SectionStringTable + Sec.sh_name);
    if (Name == ".text") {
      TextSectionOffset = Sec.sh_offset;
      TextSectionSize = Sec.sh_size;
      TextSectionIndex = I;

      if (Verbose) {
        errs() << "  Found .text section at offset "
               << format_hex(TextSectionOffset, 8) << " size "
               << TextSectionSize << "\n";
      }
      return Error::success();
    }
  }

  return createStringError(inconvertibleErrorCode(),
                           "No .text section found in input file");
}

uint32_t ELFRewriter::computeNewFlags(uint32_t OldFlags) {
  // The machine type is in the lower 8 bits of e_flags for AMDGPU
  // Mask: EF_AMDGPU_MACH = 0xff
  const uint32_t EF_AMDGPU_MACH_MASK = 0xff;

  unsigned NewMach = getAMDGPUElfMach(TargetCPU);
  if (NewMach == 0) {
    // Unknown target, keep original flags
    if (Verbose)
      errs() << "  Warning: Unknown target architecture " << TargetCPU
             << ", keeping original flags\n";
    return OldFlags;
  }

  // Replace the machine type, keep other flags (xnack, sramecc, etc.)
  uint32_t NewFlags = (OldFlags & ~EF_AMDGPU_MACH_MASK) | NewMach;

  if (Verbose) {
    errs() << "  Updating e_flags: " << format_hex(OldFlags, 8) << " -> "
           << format_hex(NewFlags, 8) << "\n";
  }

  return NewFlags;
}

std::string ELFRewriter::buildTargetString() const {
  // Build the target string in the format expected by AMDGPU HSA metadata.
  // Format: "amdgcn-amd-amdhsa--<gfxNNN>"
  // For example: "amdgcn-amd-amdhsa--gfx90a"
  return "amdgcn-amd-amdhsa--" + TargetCPU.str();
}

Error ELFRewriter::updateNoteSections() {
  // Find and update .note sections containing AMDGPU metadata.
  // The metadata is in MsgPack format and contains "amdhsa.target" field.

  for (uint16_t I = 0; I < NumSections; ++I) {
    const Elf64_Shdr &Sec = SectionHeaders[I];

    // Only process SHT_NOTE sections
    if (Sec.sh_type != ELF::SHT_NOTE)
      continue;

    // Verify section bounds
    if (Sec.sh_offset + Sec.sh_size > InputSize)
      continue;

    // Parse notes in this section
    uint64_t Offset = 0;
    const uint8_t *SectionData = reinterpret_cast<const uint8_t *>(
        OutputBuffer.data() + Sec.sh_offset);

    while (Offset + 12 <= Sec.sh_size) {
      // Note header: namesz (4), descsz (4), type (4)
      uint32_t NameSz = support::endian::read32le(SectionData + Offset);
      uint32_t DescSz = support::endian::read32le(SectionData + Offset + 4);
      uint32_t Type = support::endian::read32le(SectionData + Offset + 8);

      // Align to 4 bytes
      uint32_t NameSzAligned = (NameSz + 3) & ~3u;
      uint32_t DescSzAligned = (DescSz + 3) & ~3u;

      // Check bounds
      uint64_t NoteEnd = Offset + 12 + NameSzAligned + DescSzAligned;
      if (NoteEnd > Sec.sh_size)
        break;

      // Check if this is an AMDGPU metadata note
      if (Type == ELF::NT_AMDGPU_METADATA && NameSz > 0) {
        StringRef Name(reinterpret_cast<const char *>(SectionData + Offset + 12),
                       NameSz - 1); // Exclude null terminator

        if (Name == "AMDGPU") {
          // Found AMDGPU metadata note - parse and update the MsgPack content
          const uint8_t *DescStart = SectionData + Offset + 12 + NameSzAligned;
          StringRef MsgPackBlob(reinterpret_cast<const char *>(DescStart),
                                DescSz);

          msgpack::Document Doc;
          if (!Doc.readFromBlob(MsgPackBlob, /*Multi=*/false)) {
            if (Verbose)
              errs() << "  Warning: Failed to parse AMDGPU MsgPack metadata\n";
          } else {
            // Update the amdhsa.target field
            DocNode &Root = Doc.getRoot();
            if (Root.isMap()) {
              MapDocNode &RootMap = Root.getMap();
              auto It = RootMap.find("amdhsa.target");
              if (It != RootMap.end()) {
                std::string OldTarget;
                if (It->second.isString())
                  OldTarget = It->second.getString().str();

                // Update to new target
                std::string NewTarget = buildTargetString();
                It->second = Doc.getNode(NewTarget, /*Copy=*/true);

                if (Verbose) {
                  errs() << "  Updating amdhsa.target: \"" << OldTarget
                         << "\" -> \"" << NewTarget << "\"\n";
                }

                // Serialize back to MsgPack
                std::string NewBlob;
                Doc.writeToBlob(NewBlob);

                // Check if the new blob fits in the existing space
                if (NewBlob.size() <= DescSz) {
                  // Copy the new blob and pad with zeros
                  uint8_t *DescDst = OutputBuffer.data() + Sec.sh_offset +
                                     Offset + 12 + NameSzAligned;
                  std::memcpy(DescDst, NewBlob.data(), NewBlob.size());
                  std::memset(DescDst + NewBlob.size(), 0,
                              DescSz - NewBlob.size());
                } else {
                  // New blob is larger - this is rare since target strings
                  // are typically the same length (e.g., "gfx942" -> "gfx90a")
                  // For now, warn and continue - a more robust solution would
                  // require section resizing
                  errs() << "  Warning: New MsgPack metadata is larger than "
                            "original ("
                         << NewBlob.size() << " vs " << DescSz
                         << "). Metadata not updated.\n";
                }
              }
            }
          }
        }
      }

      Offset = NoteEnd;
    }
  }

  return Error::success();
}

Error ELFRewriter::buildOutput(ArrayRef<char> NewTextSection) {
  // Calculate size difference
  int64_t SizeDiff =
      static_cast<int64_t>(NewTextSection.size()) - TextSectionSize;

  if (SizeDiff <= 0) {
    // Same size or smaller - simple in-place replacement
    OutputBuffer.assign(InputData, InputData + InputSize);

    // Patch the e_flags in the ELF header
    Elf64_Ehdr *OutHeader = reinterpret_cast<Elf64_Ehdr *>(OutputBuffer.data());
    OutHeader->e_flags = computeNewFlags(Header->e_flags);

    // Patch the .text section content
    std::memcpy(OutputBuffer.data() + TextSectionOffset, NewTextSection.data(),
                NewTextSection.size());

    if (SizeDiff < 0) {
      // Pad remaining space with s_nop 0 instructions
      uint32_t SNop = 0xBF800000; // s_nop 0
      size_t PadStart = TextSectionOffset + NewTextSection.size();
      size_t PadEnd = TextSectionOffset + TextSectionSize;

      for (size_t Off = PadStart; Off + 4 <= PadEnd; Off += 4) {
        std::memcpy(OutputBuffer.data() + Off, &SNop, 4);
      }

      if (Verbose) {
        errs() << "  Padded " << (PadEnd - PadStart)
               << " bytes with s_nop instructions\n";
      }
    }
  } else {
    // Section expansion required
    if (Verbose) {
      errs() << "  Section expansion needed: " << TextSectionSize << " -> "
             << NewTextSection.size() << " (+" << SizeDiff << " bytes)\n";
    }

    // Align size difference to preserve alignment (typically 16 or 256 bytes)
    // Use 256-byte alignment for AMDGPU code sections
    const uint64_t TextAlignment = 256;
    uint64_t AlignedSizeDiff = (SizeDiff + TextAlignment - 1) & ~(TextAlignment - 1);

    // Calculate new file size
    size_t NewSize = InputSize + AlignedSizeDiff;
    OutputBuffer.resize(NewSize);

    // Copy everything before .text section
    std::memcpy(OutputBuffer.data(), InputData, TextSectionOffset);

    // Copy new .text content
    std::memcpy(OutputBuffer.data() + TextSectionOffset, NewTextSection.data(),
                NewTextSection.size());

    // Pad to alignment
    if (NewTextSection.size() < TextSectionSize + AlignedSizeDiff) {
      uint32_t SNop = 0xBF800000; // s_nop 0
      size_t PadStart = TextSectionOffset + NewTextSection.size();
      size_t PadEnd = TextSectionOffset + TextSectionSize + AlignedSizeDiff;
      for (size_t Off = PadStart; Off + 4 <= PadEnd; Off += 4) {
        std::memcpy(OutputBuffer.data() + Off, &SNop, 4);
      }
    }

    // Copy everything after .text section, shifted by AlignedSizeDiff
    uint64_t TextEnd = TextSectionOffset + TextSectionSize;
    uint64_t RemainingSize = InputSize - TextEnd;
    if (RemainingSize > 0) {
      std::memcpy(OutputBuffer.data() + TextSectionOffset + TextSectionSize + AlignedSizeDiff,
                  InputData + TextEnd, RemainingSize);
    }

    // Update ELF header
    Elf64_Ehdr *OutHeader = reinterpret_cast<Elf64_Ehdr *>(OutputBuffer.data());
    OutHeader->e_flags = computeNewFlags(Header->e_flags);

    // Update e_shoff if section headers are after .text
    if (Header->e_shoff >= TextEnd) {
      OutHeader->e_shoff = Header->e_shoff + AlignedSizeDiff;
    }

    // Get pointers to new section headers
    Elf64_Shdr *OutSectionHeaders =
        reinterpret_cast<Elf64_Shdr *>(OutputBuffer.data() + OutHeader->e_shoff);

    // Update section headers
    for (uint16_t I = 0; I < NumSections; ++I) {
      Elf64_Shdr &OutSec = OutSectionHeaders[I];
      const Elf64_Shdr &InSec = SectionHeaders[I];

      // Copy original section header
      OutSec = InSec;

      // Update .text section size
      if (I == TextSectionIndex) {
        OutSec.sh_size = TextSectionSize + AlignedSizeDiff;
        if (Verbose) {
          errs() << "  Updated .text section size: " << OutSec.sh_size << "\n";
        }
      }
      // Shift sections that come after .text
      else if (InSec.sh_offset >= TextEnd) {
        OutSec.sh_offset = InSec.sh_offset + AlignedSizeDiff;
      }
    }

    // Update program headers (segments)
    if (ProgramHeaders && NumProgramHeaders > 0) {
      Elf64_Phdr *OutProgramHeaders =
          reinterpret_cast<Elf64_Phdr *>(OutputBuffer.data() + Header->e_phoff);

      for (uint16_t I = 0; I < NumProgramHeaders; ++I) {
        Elf64_Phdr &OutPhdr = OutProgramHeaders[I];
        const Elf64_Phdr &InPhdr = ProgramHeaders[I];

        // Copy original program header
        OutPhdr = InPhdr;

        // Check if this segment contains .text
        if (InPhdr.p_type == ELF::PT_LOAD &&
            TextSectionOffset >= InPhdr.p_offset &&
            TextSectionOffset < InPhdr.p_offset + InPhdr.p_filesz) {
          // This segment contains .text - expand it
          OutPhdr.p_filesz = InPhdr.p_filesz + AlignedSizeDiff;
          OutPhdr.p_memsz = InPhdr.p_memsz + AlignedSizeDiff;
          if (Verbose) {
            errs() << "  Updated segment " << I << " sizes: filesz="
                   << OutPhdr.p_filesz << " memsz=" << OutPhdr.p_memsz << "\n";
          }
        }
        // Segments starting after .text need their offset shifted
        else if (InPhdr.p_offset >= TextEnd) {
          OutPhdr.p_offset = InPhdr.p_offset + AlignedSizeDiff;
        }
      }
    }

    // Adjust relocation entries
    if (auto Err = adjustRelocations(TextEnd, AlignedSizeDiff))
      return Err;
  }

  // Update .note.AMD sections with new architecture info
  if (auto Err = updateNoteSections())
    return Err;

  return Error::success();
}

Error ELFRewriter::adjustRelocations(uint64_t TextEnd, uint64_t AlignedSizeDiff) {
  // Adjust relocation entries when section offsets change.
  // This is needed when .text expansion causes following sections to shift.
  //
  // For AMDGPU code objects, most relocations are:
  // 1. Within .text (branch targets) - these don't need adjustment as they
  //    are relative to the section, not absolute
  // 2. External symbols - these also don't need adjustment
  //
  // However, relocations referencing data in sections after .text may need
  // r_offset adjustment if those sections have moved.

  Elf64_Ehdr *OutHeader = reinterpret_cast<Elf64_Ehdr *>(OutputBuffer.data());
  Elf64_Shdr *OutSectionHeaders =
      reinterpret_cast<Elf64_Shdr *>(OutputBuffer.data() + OutHeader->e_shoff);

  for (uint16_t I = 0; I < NumSections; ++I) {
    Elf64_Shdr &Sec = OutSectionHeaders[I];

    // Process SHT_RELA and SHT_REL sections
    if (Sec.sh_type != ELF::SHT_RELA && Sec.sh_type != ELF::SHT_REL)
      continue;

    // Get the section these relocations apply to
    uint32_t TargetSectionIdx = Sec.sh_info;
    if (TargetSectionIdx >= NumSections)
      continue;

    // Determine if we need to adjust based on target section
    // If the target section was shifted, we may need to update r_offset values
    // However, for most AMDGPU relocations, r_offset is section-relative
    // and doesn't need adjustment.

    // Check if the relocation section itself was shifted
    const Elf64_Shdr &OrigSec = SectionHeaders[I];
    if (OrigSec.sh_offset < TextEnd)
      continue; // Relocation section is before .text, no adjustment needed

    // The relocation section was shifted - verify the data is at the new offset
    // (This is handled by buildOutput's section shifting logic)

    // Process relocations to update r_offset if needed
    if (Sec.sh_type == ELF::SHT_RELA) {
      // RELA entries have explicit addends
      using Elf64_Rela = ELF::Elf64_Rela;
      size_t NumRelocs = Sec.sh_size / sizeof(Elf64_Rela);
      Elf64_Rela *Relocs =
          reinterpret_cast<Elf64_Rela *>(OutputBuffer.data() + Sec.sh_offset);

      for (size_t J = 0; J < NumRelocs; ++J) {
        // r_offset is typically section-relative for AMDGPU
        // However, if it's an absolute address that falls in a shifted region,
        // we need to adjust it.
        //
        // For most AMDGPU relocations, this adjustment isn't needed because:
        // 1. Code relocations are relative
        // 2. Data relocations point to symbols, not absolute addresses
        //
        // But we handle the case where r_offset is an absolute file offset
        // pointing to a shifted section.
        if (Relocs[J].r_offset >= TextEnd) {
          Relocs[J].r_offset += AlignedSizeDiff;
        }
      }
    } else {
      // REL entries (addend in target location)
      using Elf64_Rel = ELF::Elf64_Rel;
      size_t NumRelocs = Sec.sh_size / sizeof(Elf64_Rel);
      Elf64_Rel *Relocs =
          reinterpret_cast<Elf64_Rel *>(OutputBuffer.data() + Sec.sh_offset);

      for (size_t J = 0; J < NumRelocs; ++J) {
        if (Relocs[J].r_offset >= TextEnd) {
          Relocs[J].r_offset += AlignedSizeDiff;
        }
      }
    }

    if (Verbose) {
      errs() << "  Adjusted relocations in section " << I << "\n";
    }
  }

  return Error::success();
}

Error ELFRewriter::writeOutput(StringRef OutputPath) {
  std::error_code EC;
  raw_fd_ostream OutFile(OutputPath, EC, sys::fs::OF_None);
  if (EC)
    return createStringError(EC, "Failed to open output file: " + OutputPath);

  OutFile.write(reinterpret_cast<const char *>(OutputBuffer.data()),
                OutputBuffer.size());

  if (OutFile.has_error())
    return createStringError(OutFile.error(),
                             "Failed to write output file: " + OutputPath);

  return Error::success();
}

Error ELFRewriter::rewrite(ArrayRef<char> NewTextSection, StringRef OutputPath) {
  if (auto Err = parseELFHeader())
    return Err;

  if (auto Err = findTextSection())
    return Err;

  if (auto Err = buildOutput(NewTextSection))
    return Err;

  if (auto Err = writeOutput(OutputPath))
    return Err;

  return Error::success();
}

} // anonymous namespace

Error llvm::amdgpu::rewriteELFWithNewText(const MemoryBuffer &InputBuffer,
                                          ArrayRef<char> NewTextSection,
                                          StringRef SourceCPU,
                                          StringRef TargetCPU,
                                          StringRef OutputPath,
                                          bool Verbose) {
  ELFRewriter Rewriter(InputBuffer, SourceCPU, TargetCPU, Verbose);
  return Rewriter.rewrite(NewTextSection, OutputPath);
}
