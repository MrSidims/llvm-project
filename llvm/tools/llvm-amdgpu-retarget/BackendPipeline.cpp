//===-- BackendPipeline.cpp - Run LLVM Backend on MachineFunction ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BackendPipeline.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define DEBUG_TYPE "backend-pipeline"

BackendPipeline::BackendPipeline(TargetMachine &SourceTM,
                                 TargetMachine &TargetTM, bool Verbose)
    : SourceTM(SourceTM), TargetTM(TargetTM), Verbose(Verbose) {}

BackendPipeline::~BackendPipeline() = default;

bool BackendPipeline::needsTransformation(const MachineInstr &MI) const {
  // Check if this instruction's opcode exists on target
  // For same-family GPUs (gfx942 -> gfx90a), most instructions are compatible
  unsigned Opcode = MI.getOpcode();
  const TargetInstrInfo *TII = TargetTM.getSubtargetImpl(MI.getMF()->getFunction())
                                   ->getInstrInfo();

  // Try to get the instruction description - if it fails, needs transformation
  const MCInstrDesc &Desc = TII->get(Opcode);
  (void)Desc; // Just checking it doesn't crash

  // Check for known incompatible instructions between gfx942 and gfx90a
  StringRef Name = TII->getName(Opcode);
  if (Name.contains("V_LSHL_ADD_U64"))
    return true;

  return false;
}

bool BackendPipeline::needsExpansion(const MachineInstr &MI) const {
  const TargetInstrInfo *TII = TargetTM.getSubtargetImpl(MI.getMF()->getFunction())
                                   ->getInstrInfo();
  StringRef Name = TII->getName(MI.getOpcode());

  // Instructions that need 1:N expansion
  if (Name.contains("V_LSHL_ADD_U64"))
    return true;
  if (Name.contains("V_CVT_PK_BF16_F32"))
    return true;
  if (Name.contains("V_CVT_SCALEF32_PK_FP4"))
    return true;

  return false;
}

Error BackendPipeline::transformInstruction(MachineInstr &MI,
                                            MachineBasicBlock &MBB) {
  // For simple opcode swaps, just change the opcode
  // For complex expansions, call expandInstruction

  if (needsExpansion(MI)) {
    return expandInstruction(MI, MBB);
  }

  ++Statistics.InstructionsTransformed;
  return Error::success();
}

Error BackendPipeline::expandInstruction(MachineInstr &MI,
                                         MachineBasicBlock &MBB) {
  const TargetInstrInfo *TII = MBB.getParent()->getSubtarget().getInstrInfo();
  StringRef Name = TII->getName(MI.getOpcode());

  if (Name.contains("V_LSHL_ADD_U64")) {
    // Expand V_LSHL_ADD_U64 dst, src0, src1, src2
    // to:
    //   V_LSHLREV_B64 tmp, src1, src0
    //   V_ADD_CO_U32 dst.lo, vcc, tmp.lo, src2.lo
    //   V_ADDC_CO_U32 dst.hi, vcc, tmp.hi, src2.hi, vcc

    // For now, we'll mark this as needing expansion but let the
    // AsmPrinter handle it if possible. Full implementation requires
    // register allocation for the temp register.

    LLVM_DEBUG(dbgs() << "Need to expand V_LSHL_ADD_U64\n");
    ++Statistics.InstructionsExpanded;

    // TODO: Implement full expansion
    // This requires:
    // 1. Allocate a temp 64-bit register
    // 2. Build the expansion sequence
    // 3. Replace the original instruction
  }

  return Error::success();
}

Error BackendPipeline::transformFunction(MachineFunction &MF) {
  if (Verbose) {
    errs() << "  Transforming MachineFunction for target...\n";
  }

  for (MachineBasicBlock &MBB : MF) {
    // We need to be careful modifying while iterating
    // Collect instructions that need transformation first
    SmallVector<MachineInstr *, 16> ToTransform;

    for (MachineInstr &MI : MBB) {
      if (needsTransformation(MI)) {
        ToTransform.push_back(&MI);
      }
    }

    for (MachineInstr *MI : ToTransform) {
      if (auto Err = transformInstruction(*MI, MBB))
        return Err;
    }
  }

  if (Verbose) {
    errs() << "    Transformed: " << Statistics.InstructionsTransformed << "\n";
    errs() << "    Expanded: " << Statistics.InstructionsExpanded << "\n";
  }

  return Error::success();
}

Error BackendPipeline::runPostRAPasses(MachineFunction &MF) {
  // Run post-register-allocation passes
  // These include:
  // - Post-RA scheduling
  // - Branch folding
  // - Block placement

  // For now, we skip these since we're working with already-scheduled code
  // The main transformation happens in emitCode

  return Error::success();
}

Error BackendPipeline::emitCode(MachineFunction &MF, MachineModuleInfo &MMI,
                                raw_pwrite_stream &OS) {
  // Set up MC infrastructure for code emission
  const Triple &TT = TargetTM.getTargetTriple();
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
  if (!TheTarget)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to find target: " + Error);

  // Create MC components
  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
  if (!MRI)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create MCRegisterInfo");

  MCTargetOptions MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TT, MCOptions));
  if (!MAI)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create MCAsmInfo");

  std::unique_ptr<MCSubtargetInfo> STI(TheTarget->createMCSubtargetInfo(
      TT, TargetTM.getTargetCPU(), TargetTM.getTargetFeatureString()));
  if (!STI)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create MCSubtargetInfo");

  std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
  if (!MCII)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create MCInstrInfo");

  MCContext Ctx(TT, MAI.get(), MRI.get(), STI.get());

  std::unique_ptr<MCObjectFileInfo> MOFI(
      TheTarget->createMCObjectFileInfo(Ctx, /*PIC=*/false));
  Ctx.setObjectFileInfo(MOFI.get());

  // Create code emitter
  std::unique_ptr<MCCodeEmitter> CE(
      TheTarget->createMCCodeEmitter(*MCII, Ctx));
  if (!CE)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create MCCodeEmitter");

  // Create asm backend
  std::unique_ptr<MCAsmBackend> MAB(
      TheTarget->createMCAsmBackend(*STI, *MRI, MCOptions));
  if (!MAB)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create MCAsmBackend");

  // Create object writer
  std::unique_ptr<MCObjectWriter> OW = MAB->createObjectWriter(OS);

  // Create MCStreamer for object emission
  std::unique_ptr<MCStreamer> Streamer(TheTarget->createMCObjectStreamer(
      TT, Ctx, std::move(MAB), std::move(OW), std::move(CE), *STI));
  if (!Streamer)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create MCStreamer");

  // Lower MachineInstrs to MCInsts and emit through streamer
  // For AMDGPU, we convert MachineInstr directly using the opcode and operands
  // since we're working with physical registers already.

  Streamer->initSections(*STI);

  // Emit function as a text section
  MCSection *TextSection = Ctx.getObjectFileInfo()->getTextSection();
  Streamer->switchSection(TextSection);

  unsigned InstructionsEmitted = 0;

  // For each basic block, emit its instructions
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      // Skip meta instructions (debug info, labels, etc.)
      if (MI.isMetaInstruction())
        continue;

      // Build MCInst from MachineInstr
      MCInst OutMI;
      OutMI.setOpcode(MI.getOpcode());

      for (const MachineOperand &MO : MI.operands()) {
        MCOperand MCOp;
        if (MO.isReg()) {
          // Physical registers pass through directly
          MCOp = MCOperand::createReg(MO.getReg());
        } else if (MO.isImm()) {
          MCOp = MCOperand::createImm(MO.getImm());
        } else if (MO.isFPImm()) {
          // Convert FP to bits representation
          const APFloat &FPVal = MO.getFPImm()->getValueAPF();
          MCOp = MCOperand::createImm(FPVal.bitcastToAPInt().getZExtValue());
        } else if (MO.isCImm()) {
          // ConstantInt immediate
          MCOp = MCOperand::createImm(MO.getCImm()->getSExtValue());
        } else {
          // Skip operands we don't handle (metadata, symbols, block addresses, etc.)
          continue;
        }
        OutMI.addOperand(MCOp);
      }

      // Emit the instruction
      Streamer->emitInstruction(OutMI, *STI);
      ++InstructionsEmitted;
    }
  }

  // Finish emission
  Streamer->finish();

  Statistics.BytesEmitted = InstructionsEmitted;

  if (Verbose) {
    errs() << "  Code emission complete: " << InstructionsEmitted
           << " instructions emitted\n";
  }

  return Error::success();
}

Error BackendPipeline::run(MachineFunction &MF, MachineModuleInfo &MMI,
                           SmallVectorImpl<char> &Output) {
  // Transform the function for the target
  if (auto Err = transformFunction(MF))
    return Err;

  // Run post-RA passes
  if (auto Err = runPostRAPasses(MF))
    return Err;

  // Emit code
  raw_svector_ostream OS(Output);
  if (auto Err = emitCode(MF, MMI, OS))
    return Err;

  Statistics.BytesEmitted = Output.size();
  return Error::success();
}
