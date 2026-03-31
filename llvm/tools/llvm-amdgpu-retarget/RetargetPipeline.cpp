//===-- RetargetPipeline.cpp - Full MCInst->MIR->Code Pipeline ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RetargetPipeline.h"
#include "BackendPipeline.h"
#include "ELFRetargetWriter.h"
#include "MIRLifter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::object;

#define DEBUG_TYPE "retarget-pipeline"

RetargetPipeline::RetargetPipeline(StringRef SourceCPU, StringRef TargetCPU,
                                   bool Verbose)
    : SourceCPU(SourceCPU.str()), TargetCPU(TargetCPU.str()), Verbose(Verbose) {
  Ctx = std::make_unique<LLVMContext>();
}

RetargetPipeline::~RetargetPipeline() = default;

Error RetargetPipeline::initialize() {
  // Initialize AMDGPU target
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();

  return setupTargets();
}

Error RetargetPipeline::setupTargets() {
  // Set up triple for AMDGPU HSA
  TheTriple.setArch(Triple::amdgcn);
  TheTriple.setVendor(Triple::AMD);
  TheTriple.setOS(Triple::AMDHSA);

  std::string Error;
  TheTarget = TargetRegistry::lookupTarget(TheTriple, Error);
  if (!TheTarget)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to find AMDGPU target: " + Error);

  // Create TargetMachine for source architecture
  TargetOptions Options;
  auto RM = std::optional<Reloc::Model>(Reloc::PIC_);

  SourceTM.reset(TheTarget->createTargetMachine(
      TheTriple, SourceCPU, "", Options, RM, std::nullopt,
      CodeGenOptLevel::Default));
  if (!SourceTM)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create TargetMachine for " + SourceCPU);

  // Create TargetMachine for target architecture
  TargetTM.reset(TheTarget->createTargetMachine(
      TheTriple, TargetCPU, "", Options, RM, std::nullopt,
      CodeGenOptLevel::Default));
  if (!TargetTM)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create TargetMachine for " + TargetCPU);

  // Set up MC components for disassembly
  SourceMRI.reset(TheTarget->createMCRegInfo(TheTriple));
  if (!SourceMRI)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create MCRegisterInfo");

  MCTargetOptions MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*SourceMRI, TheTriple, MCOptions));
  if (!MAI)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create MCAsmInfo");

  SourceSTI.reset(
      TheTarget->createMCSubtargetInfo(TheTriple, SourceCPU, ""));
  if (!SourceSTI)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create MCSubtargetInfo");

  SourceMCII.reset(TheTarget->createMCInstrInfo());
  if (!SourceMCII)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create MCInstrInfo");

  // Create MCContext for disassembler
  auto MCCtx = std::make_unique<MCContext>(TheTriple, MAI.get(), SourceMRI.get(),
                                           SourceSTI.get());

  SourceDisasm.reset(TheTarget->createMCDisassembler(*SourceSTI, *MCCtx));
  if (!SourceDisasm)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create MCDisassembler");

  if (Verbose) {
    errs() << "Initialized pipeline:\n"
           << "  Source: " << SourceCPU << "\n"
           << "  Target: " << TargetCPU << "\n";
  }

  return Error::success();
}

Error RetargetPipeline::disassemble(ArrayRef<uint8_t> TextSection,
                                    SmallVectorImpl<MCInst> &Instructions,
                                    SmallVectorImpl<uint64_t> &Offsets) {
  uint64_t Offset = 0;
  while (Offset < TextSection.size()) {
    MCInst Inst;
    uint64_t Size;
    ArrayRef<uint8_t> InstBytes = TextSection.slice(Offset);

    MCDisassembler::DecodeStatus Status =
        SourceDisasm->getInstruction(Inst, Size, InstBytes, Offset, nulls());

    if (Status == MCDisassembler::Fail) {
      return createStringError(inconvertibleErrorCode(),
                               "Failed to disassemble at offset " +
                                   Twine::utohexstr(Offset));
    }

    Instructions.push_back(Inst);
    Offsets.push_back(Offset);
    Offset += Size;
  }

  Statistics.NumInstructions = Instructions.size();

  if (Verbose) {
    errs() << "  Disassembled " << Instructions.size() << " instructions\n";
  }

  return Error::success();
}

Error RetargetPipeline::liftToMIR(ArrayRef<MCInst> Instructions,
                                  ArrayRef<uint64_t> Offsets,
                                  StringRef FunctionName) {
  // Create a MIRLifter and lift the instructions
  // Store the lifter as a member so it stays alive
  Lifter = std::make_unique<MIRLifter>(*Ctx, *SourceTM, *SourceMCII, *SourceMRI, Verbose);

  auto MFOrErr = Lifter->lift(Instructions, Offsets, FunctionName);
  if (!MFOrErr)
    return MFOrErr.takeError();

  CurrentMF = *MFOrErr;
  Statistics.NumBasicBlocks = CurrentMF->size();

  if (Verbose) {
    errs() << "  Lifted to MachineFunction with " << CurrentMF->size()
           << " basic blocks\n";
  }

  return Error::success();
}

Error RetargetPipeline::transformForTarget() {
  if (!CurrentMF)
    return createStringError(inconvertibleErrorCode(),
                             "No MachineFunction to transform");

  // Create backend pipeline for transformation
  BackendPipeline Backend(*SourceTM, *TargetTM, Verbose);

  if (Verbose) {
    errs() << "  Transforming for target architecture...\n";
  }

  // Transform the MachineFunction for the target
  if (auto Err = Backend.transformFunction(*CurrentMF))
    return Err;

  const auto &Stats = Backend.getStats();
  Statistics.NumTransformed = Stats.InstructionsTransformed;
  Statistics.NumExpanded = Stats.InstructionsExpanded;

  return Error::success();
}

Error RetargetPipeline::emitCode(SmallVectorImpl<char> &Output) {
  if (!CurrentMF || !Lifter)
    return createStringError(inconvertibleErrorCode(),
                             "No MachineFunction to emit");

  if (Verbose) {
    errs() << "  Emitting code...\n";
  }

  // Create backend pipeline for code emission
  BackendPipeline Backend(*SourceTM, *TargetTM, Verbose);

  // Emit code using the backend pipeline
  raw_svector_ostream OS(Output);
  if (auto Err = Backend.emitCode(*CurrentMF, Lifter->getMMI(), OS))
    return Err;

  return Error::success();
}

Error RetargetPipeline::run(const MemoryBuffer &InputBuffer,
                            StringRef OutputPath) {
  // Parse ELF and find .text section
  Expected<std::unique_ptr<ObjectFile>> ObjOrErr =
      ObjectFile::createObjectFile(InputBuffer.getMemBufferRef());
  if (!ObjOrErr)
    return ObjOrErr.takeError();

  ObjectFile *Obj = ObjOrErr->get();
  if (!Obj->isELF())
    return createStringError(inconvertibleErrorCode(),
                             "Input is not an ELF file");

  // Find .text section
  for (const SectionRef &Section : Obj->sections()) {
    Expected<StringRef> NameOrErr = Section.getName();
    if (!NameOrErr)
      continue;

    if (*NameOrErr != ".text")
      continue;

    Expected<StringRef> ContentsOrErr = Section.getContents();
    if (!ContentsOrErr)
      return ContentsOrErr.takeError();

    ArrayRef<uint8_t> TextBytes(
        reinterpret_cast<const uint8_t *>(ContentsOrErr->data()),
        ContentsOrErr->size());

    Statistics.InputBytes = TextBytes.size();

    // Disassemble
    SmallVector<MCInst, 256> Instructions;
    SmallVector<uint64_t, 256> Offsets;
    if (auto Err = disassemble(TextBytes, Instructions, Offsets))
      return Err;

    // Lift to MIR
    if (auto Err = liftToMIR(Instructions, Offsets, "retargeted_kernel"))
      return Err;

    // Transform for target
    if (auto Err = transformForTarget())
      return Err;

    // Emit code - for now we'll use the ELFRetargetWriter like the original pipeline
    // The MIR pipeline generates proper code but we need to extract just the .text
    // section and rewrite the original ELF with updated flags/metadata.
    //
    // For a complete implementation, we would either:
    // 1. Extract just the .text bytes from the emitted object
    // 2. Use the ELFRetargetWriter to rewrite the original ELF
    //
    // For now, emit to a temp buffer to verify the pipeline works
    SmallVector<char, 4096> Output;
    if (auto Err = emitCode(Output))
      return Err;

    Statistics.OutputBytes = Output.size();

    // Write the emitted object file directly
    // Note: This writes a complete ELF, not just the retargeted .text
    if (!OutputPath.empty()) {
      std::error_code EC;
      raw_fd_ostream OutFile(OutputPath, EC, sys::fs::OF_None);
      if (EC)
        return createStringError(EC, "Failed to open output file: " + OutputPath);
      OutFile.write(Output.data(), Output.size());
    }

    break;
  }

  return Error::success();
}
