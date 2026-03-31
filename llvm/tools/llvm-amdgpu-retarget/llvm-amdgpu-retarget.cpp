//===-- llvm-amdgpu-retarget.cpp - AMDGPU Binary Retargeting Tool ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility retargets AMDGPU code objects from one GPU architecture to
// another. It enables running binaries compiled for newer GPUs (e.g., gfx950)
// on older compatible GPUs (e.g., gfx942) by translating instructions.
//
// Usage:
//   llvm-amdgpu-retarget --source=gfx950 --target=gfx942 input.co -o output.co
//
// This tool:
// 1. Parses the input ELF using LLVM's object library
// 2. Disassembles .text using LLVM MC (source ISA decoder)
// 3. Transforms MCInsts using instruction mapping tables
// 4. For emulation sequences: generates equivalent instruction sequences
// 5. Reassembles using LLVM MC (target ISA encoder)
// 6. Rewrites ELF metadata and outputs a new .co file
//
//===----------------------------------------------------------------------===//

#include "AMDGPURetargeter.h"
#include "ELFRetargetWriter.h"
#include "RetargetPipeline.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <memory>

using namespace llvm;
using namespace llvm::object;

static cl::OptionCategory RetargetCategory("Retarget Options");

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::Required,
                                          cl::cat(RetargetCategory));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(RetargetCategory));

static cl::opt<std::string> SourceArch("source",
                                       cl::desc("Source GPU architecture"),
                                       cl::value_desc("arch"),
                                       cl::Required,
                                       cl::cat(RetargetCategory));

static cl::opt<std::string> TargetArch("target",
                                       cl::desc("Target GPU architecture"),
                                       cl::value_desc("arch"),
                                       cl::Required,
                                       cl::cat(RetargetCategory));

static cl::opt<bool> Verbose("v", cl::desc("Verbose output"),
                             cl::cat(RetargetCategory));

static cl::opt<bool> DryRun("dry-run",
                            cl::desc("Analyze without producing output"),
                            cl::cat(RetargetCategory));

static cl::opt<bool> UseMIRPipeline("use-mir",
                                    cl::desc("Use MIR-based pipeline (experimental)"),
                                    cl::cat(RetargetCategory));

namespace {

class AMDGPURetargetTool {
public:
  AMDGPURetargetTool(StringRef SourceCPU, StringRef TargetCPU)
      : SourceCPU(SourceCPU), TargetCPU(TargetCPU) {}

  Error initialize();
  Error processFile(StringRef InputPath, StringRef OutputPath);

private:
  Error setupMCComponents(StringRef CPU, std::unique_ptr<MCContext> &Ctx,
                          std::unique_ptr<MCSubtargetInfo> &STI,
                          std::unique_ptr<MCDisassembler> &Disasm,
                          std::unique_ptr<MCInstrInfo> &MCII,
                          std::unique_ptr<MCRegisterInfo> &MRI,
                          std::unique_ptr<MCAsmInfo> &MAI,
                          std::unique_ptr<MCCodeEmitter> &CE);

  Error disassembleSection(const SectionRef &Section,
                           const MCDisassembler &Disasm,
                           const MCInstrInfo &MCII,
                           SmallVectorImpl<MCInst> &Instructions,
                           SmallVectorImpl<uint64_t> &Offsets);

  Error transformInstructions(ArrayRef<MCInst> SourceInsts,
                              ArrayRef<uint64_t> Offsets,
                              SmallVectorImpl<MCInst> &TargetInsts);

  Error encodeInstructions(ArrayRef<MCInst> Instructions,
                           const MCCodeEmitter &CE,
                           const MCSubtargetInfo &STI,
                           SmallVectorImpl<char> &Output);

  std::string SourceCPU;
  std::string TargetCPU;
  const Target *TheTarget = nullptr;
  Triple TheTriple;

  // MC components for source architecture
  std::unique_ptr<MCContext> SourceCtx;
  std::unique_ptr<MCSubtargetInfo> SourceSTI;
  std::unique_ptr<MCDisassembler> SourceDisasm;
  std::unique_ptr<MCInstrInfo> SourceMCII;
  std::unique_ptr<MCRegisterInfo> SourceMRI;
  std::unique_ptr<MCAsmInfo> SourceMAI;
  std::unique_ptr<MCCodeEmitter> SourceCE;

  // MC components for target architecture
  std::unique_ptr<MCContext> TargetCtx;
  std::unique_ptr<MCSubtargetInfo> TargetSTI;
  std::unique_ptr<MCDisassembler> TargetDisasm;
  std::unique_ptr<MCInstrInfo> TargetMCII;
  std::unique_ptr<MCRegisterInfo> TargetMRI;
  std::unique_ptr<MCAsmInfo> TargetMAI;
  std::unique_ptr<MCCodeEmitter> TargetCE;

  std::unique_ptr<AMDGPURetargeter> Retargeter;
};

Error AMDGPURetargetTool::initialize() {
  // Initialize targets
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();

  // Set up triple for AMDGPU
  TheTriple.setArch(Triple::amdgcn);
  TheTriple.setVendor(Triple::AMD);
  TheTriple.setOS(Triple::AMDHSA);

  std::string Error;
  TheTarget = TargetRegistry::lookupTarget(TheTriple, Error);
  if (!TheTarget)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to find AMDGPU target: " + Error);

  // Initialize MC components for both architectures
  if (auto Err = setupMCComponents(SourceCPU, SourceCtx, SourceSTI, SourceDisasm,
                                   SourceMCII, SourceMRI, SourceMAI, SourceCE))
    return Err;

  if (auto Err = setupMCComponents(TargetCPU, TargetCtx, TargetSTI, TargetDisasm,
                                   TargetMCII, TargetMRI, TargetMAI, TargetCE))
    return Err;

  // Create the retargeter
  Retargeter = std::make_unique<AMDGPURetargeter>(
      SourceCPU, TargetCPU, *SourceMCII, *TargetMCII, *SourceMRI, *TargetMRI);

  return Error::success();
}

Error AMDGPURetargetTool::setupMCComponents(
    StringRef CPU, std::unique_ptr<MCContext> &Ctx,
    std::unique_ptr<MCSubtargetInfo> &STI,
    std::unique_ptr<MCDisassembler> &Disasm, std::unique_ptr<MCInstrInfo> &MCII,
    std::unique_ptr<MCRegisterInfo> &MRI, std::unique_ptr<MCAsmInfo> &MAI,
    std::unique_ptr<MCCodeEmitter> &CE) {

  MRI.reset(TheTarget->createMCRegInfo(TheTriple));
  if (!MRI)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create register info for " + CPU);

  MCTargetOptions MCOptions;
  MAI.reset(TheTarget->createMCAsmInfo(*MRI, TheTriple, MCOptions));
  if (!MAI)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create asm info for " + CPU);

  STI.reset(TheTarget->createMCSubtargetInfo(TheTriple, CPU, ""));
  if (!STI)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create subtarget info for " + CPU);

  MCII.reset(TheTarget->createMCInstrInfo());
  if (!MCII)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create instruction info for " + CPU);

  Ctx = std::make_unique<MCContext>(TheTriple, MAI.get(), MRI.get(), STI.get());

  Disasm.reset(TheTarget->createMCDisassembler(*STI, *Ctx));
  if (!Disasm)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create disassembler for " + CPU);

  CE.reset(TheTarget->createMCCodeEmitter(*MCII, *Ctx));
  if (!CE)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create code emitter for " + CPU);

  return Error::success();
}

Error AMDGPURetargetTool::disassembleSection(
    const SectionRef &Section, const MCDisassembler &Disasm,
    const MCInstrInfo &MCII, SmallVectorImpl<MCInst> &Instructions,
    SmallVectorImpl<uint64_t> &Offsets) {

  Expected<StringRef> ContentsOrErr = Section.getContents();
  if (!ContentsOrErr)
    return ContentsOrErr.takeError();

  ArrayRef<uint8_t> Bytes(
      reinterpret_cast<const uint8_t *>(ContentsOrErr->data()),
      ContentsOrErr->size());

  uint64_t Offset = 0;
  while (Offset < Bytes.size()) {
    MCInst Inst;
    uint64_t Size;
    ArrayRef<uint8_t> InstBytes = Bytes.slice(Offset);

    MCDisassembler::DecodeStatus Status =
        Disasm.getInstruction(Inst, Size, InstBytes, Offset, nulls());

    if (Status == MCDisassembler::Fail) {
      return createStringError(inconvertibleErrorCode(),
                               "Failed to disassemble instruction at offset " +
                                   Twine::utohexstr(Offset));
    }

    if (Status == MCDisassembler::SoftFail && Verbose) {
      errs() << "Warning: Potentially undefined instruction at offset "
             << format_hex(Offset, 8) << "\n";
    }

    Instructions.push_back(Inst);
    Offsets.push_back(Offset);
    Offset += Size;
  }

  return Error::success();
}

Error AMDGPURetargetTool::transformInstructions(
    ArrayRef<MCInst> SourceInsts, ArrayRef<uint64_t> Offsets,
    SmallVectorImpl<MCInst> &TargetInsts) {

  // Run liveness analysis for optimal register allocation
  if (auto Err = Retargeter->analyzeForLiveness(SourceInsts, Offsets))
    return Err;

  if (Verbose) {
    if (const auto *Stats = Retargeter->getLivenessStats()) {
      outs() << "  Liveness analysis: " << Stats->NumBasicBlocks << " blocks, "
             << Stats->NumIterations << " iterations\n"
             << "    Max live VGPRs: " << Stats->MaxLiveVGPRs
             << ", Min dead VGPRs: " << Stats->MinDeadVGPRs << "\n";
    }
  }

  // Transform using the new API that passes instruction index
  if (auto Err = Retargeter->transformAll(SourceInsts, Offsets, TargetInsts))
    return Err;

  return Error::success();
}

Error AMDGPURetargetTool::encodeInstructions(ArrayRef<MCInst> Instructions,
                                             const MCCodeEmitter &CE,
                                             const MCSubtargetInfo &STI,
                                             SmallVectorImpl<char> &Output) {
  SmallVector<MCFixup, 4> Fixups;

  for (const MCInst &Inst : Instructions) {
    Fixups.clear();
    CE.encodeInstruction(Inst, Output, Fixups, STI);
  }

  return Error::success();
}

Error AMDGPURetargetTool::processFile(StringRef InputPath,
                                      StringRef OutputPath) {
  // Read input file
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(InputPath);
  if (!BufferOrErr)
    return createStringError(BufferOrErr.getError(),
                             "Failed to open input file: " + InputPath);

  // Parse as ELF
  Expected<std::unique_ptr<ObjectFile>> ObjOrErr =
      ObjectFile::createObjectFile((*BufferOrErr)->getMemBufferRef());
  if (!ObjOrErr)
    return ObjOrErr.takeError();

  ObjectFile *Obj = ObjOrErr->get();
  if (!Obj->isELF())
    return createStringError(inconvertibleErrorCode(),
                             "Input is not an ELF file");

  if (Verbose)
    outs() << "Processing: " << InputPath << "\n"
           << "  Source architecture: " << SourceCPU << "\n"
           << "  Target architecture: " << TargetCPU << "\n";

  // Find and process .text section
  for (const SectionRef &Section : Obj->sections()) {
    Expected<StringRef> NameOrErr = Section.getName();
    if (!NameOrErr)
      return NameOrErr.takeError();

    if (*NameOrErr != ".text")
      continue;

    if (Verbose)
      outs() << "  Processing .text section (" << Section.getSize()
             << " bytes)\n";

    // Disassemble
    SmallVector<MCInst, 256> SourceInsts;
    SmallVector<uint64_t, 256> Offsets;
    if (auto Err = disassembleSection(Section, *SourceDisasm, *SourceMCII,
                                      SourceInsts, Offsets))
      return Err;

    if (Verbose)
      outs() << "  Disassembled " << SourceInsts.size() << " instructions\n";

    // Transform
    SmallVector<MCInst, 256> TargetInsts;
    if (auto Err = transformInstructions(SourceInsts, Offsets, TargetInsts))
      return Err;

    if (Verbose)
      outs() << "  Transformed to " << TargetInsts.size() << " instructions\n";

    if (DryRun) {
      outs() << "Dry run complete. No output file produced.\n";
      return Error::success();
    }

    // Encode
    SmallVector<char, 4096> NewText;
    if (auto Err = encodeInstructions(TargetInsts, *TargetCE, *TargetSTI, NewText))
      return Err;

    if (Verbose)
      outs() << "  Encoded " << NewText.size() << " bytes\n";

    // Get extra VGPRs needed for emulation
    unsigned ExtraVGPRs = Retargeter->getExtraVGPRsNeeded();
    if (Verbose && ExtraVGPRs > 0) {
      outs() << "  Extra VGPRs needed for emulation: " << ExtraVGPRs << "\n";
    }

    // Write output ELF with new .text section and updated flags
    if (OutputPath != "-") {
      if (auto Err = amdgpu::rewriteELFWithNewText(
              **BufferOrErr, NewText, SourceCPU, TargetCPU, OutputPath, Verbose,
              ExtraVGPRs))
        return Err;

      if (Verbose)
        outs() << "  Wrote output to " << OutputPath << "\n";
    } else {
      return createStringError(inconvertibleErrorCode(),
                               "Writing to stdout not supported for ELF output");
    }

    break;
  }

  return Error::success();
}

} // namespace

static Error runMIRPipeline(StringRef InputPath, StringRef OutputPath,
                            StringRef SourceCPU, StringRef TargetCPU) {
  // Read input file
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(InputPath);
  if (!BufferOrErr)
    return createStringError(BufferOrErr.getError(),
                             "Failed to open input file: " + InputPath);

  // Create and initialize the MIR-based pipeline
  RetargetPipeline Pipeline(SourceCPU, TargetCPU, Verbose);

  if (auto Err = Pipeline.initialize())
    return Err;

  if (auto Err = Pipeline.run(**BufferOrErr, OutputPath))
    return Err;

  // Print statistics
  const auto &Stats = Pipeline.getStats();
  if (Verbose) {
    outs() << "MIR Pipeline Statistics:\n"
           << "  Instructions: " << Stats.NumInstructions << "\n"
           << "  Basic Blocks: " << Stats.NumBasicBlocks << "\n"
           << "  Transformed: " << Stats.NumTransformed << "\n"
           << "  Expanded: " << Stats.NumExpanded << "\n"
           << "  Input bytes: " << Stats.InputBytes << "\n"
           << "  Output bytes: " << Stats.OutputBytes << "\n";
  }

  return Error::success();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions(RetargetCategory);
  cl::ParseCommandLineOptions(argc, argv,
                              "AMDGPU Binary Retargeting Tool\n\n"
                              "Retargets AMDGPU code objects from one GPU "
                              "architecture to another.\n");

  if (UseMIRPipeline) {
    // Use the experimental MIR-based pipeline
    if (auto Err = runMIRPipeline(InputFilename, OutputFilename,
                                  SourceArch, TargetArch)) {
      WithColor::error(errs(), "llvm-amdgpu-retarget")
          << toString(std::move(Err)) << "\n";
      return 1;
    }
    return 0;
  }

  // Use the standard MCInst-based pipeline
  AMDGPURetargetTool Tool(SourceArch, TargetArch);

  if (auto Err = Tool.initialize()) {
    WithColor::error(errs(), "llvm-amdgpu-retarget")
        << toString(std::move(Err)) << "\n";
    return 1;
  }

  if (auto Err = Tool.processFile(InputFilename, OutputFilename)) {
    WithColor::error(errs(), "llvm-amdgpu-retarget")
        << toString(std::move(Err)) << "\n";
    return 1;
  }

  return 0;
}
