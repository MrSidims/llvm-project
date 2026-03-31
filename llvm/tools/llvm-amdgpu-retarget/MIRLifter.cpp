//===-- MIRLifter.cpp - Lift MCInst to MachineFunction --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MIRLifter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <set>

using namespace llvm;

#define DEBUG_TYPE "mir-lifter"

MIRLifter::MIRLifter(LLVMContext &Ctx, const TargetMachine &TM,
                     const MCInstrInfo &MCII, const MCRegisterInfo &MRI,
                     bool Verbose)
    : Ctx(Ctx), TM(TM), MCII(MCII), MRI(MRI), Verbose(Verbose) {
  initializeModule();
}

MIRLifter::~MIRLifter() = default;

void MIRLifter::initializeModule() {
  TheModule = std::make_unique<Module>("lifted_module", Ctx);
  TheModule->setTargetTriple(TM.getTargetTriple());
  TheModule->setDataLayout(TM.createDataLayout());

  MMI = std::make_unique<MachineModuleInfo>(&TM);
}

bool MIRLifter::isBranch(const MCInst &Inst) const {
  const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
  return Desc.isBranch() || Desc.isCall();
}

bool MIRLifter::isConditionalBranch(const MCInst &Inst) const {
  const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
  return Desc.isConditionalBranch();
}

bool MIRLifter::isReturn(const MCInst &Inst) const {
  const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
  // For AMDGPU, s_endpgm is the return
  return Desc.isReturn() || Desc.isBarrier();
}

int64_t MIRLifter::getBranchTargetOffset(const MCInst &Inst,
                                         uint64_t InstOffset) const {
  // AMDGPU branches use PC-relative offsets: target = PC + 4 + (simm16 * 4)
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isImm()) {
      int64_t Simm = Op.getImm();
      // Check if this looks like a branch offset
      if (Simm >= -32768 && Simm <= 32767) {
        return static_cast<int64_t>(InstOffset) + 4 + (Simm * 4);
      }
    }
  }
  return -1;
}

void MIRLifter::buildCFG(ArrayRef<MCInst> Instructions,
                         ArrayRef<uint64_t> Offsets) {
  CFG.Blocks.clear();
  CFG.OffsetToBlockIndex.clear();

  if (Instructions.empty())
    return;

  // First pass: identify basic block leaders
  std::set<uint64_t> Leaders;
  Leaders.insert(Offsets[0]); // First instruction is always a leader

  for (size_t I = 0; I < Instructions.size(); ++I) {
    const MCInst &Inst = Instructions[I];
    if (isBranch(Inst)) {
      int64_t Target = getBranchTargetOffset(Inst, Offsets[I]);
      if (Target >= 0)
        Leaders.insert(static_cast<uint64_t>(Target));

      // Instruction after branch is also a leader
      if (I + 1 < Instructions.size())
        Leaders.insert(Offsets[I + 1]);

      ++Statistics.NumBranches;
    }
  }

  // Build offset to instruction index map
  DenseMap<uint64_t, size_t> OffsetToInstIdx;
  for (size_t I = 0; I < Offsets.size(); ++I)
    OffsetToInstIdx[Offsets[I]] = I;

  // Second pass: create basic blocks
  std::vector<uint64_t> SortedLeaders(Leaders.begin(), Leaders.end());

  for (size_t BI = 0; BI < SortedLeaders.size(); ++BI) {
    uint64_t LeaderOffset = SortedLeaders[BI];

    auto It = OffsetToInstIdx.find(LeaderOffset);
    if (It == OffsetToInstIdx.end())
      continue;

    size_t StartIdx = It->second;
    size_t EndIdx = Instructions.size();

    if (BI + 1 < SortedLeaders.size()) {
      auto EndIt = OffsetToInstIdx.find(SortedLeaders[BI + 1]);
      if (EndIt != OffsetToInstIdx.end())
        EndIdx = EndIt->second;
    }

    LiftedBasicBlock BB;
    BB.StartOffset = LeaderOffset;
    BB.EndOffset = (EndIdx < Offsets.size()) ? Offsets[EndIdx] : Offsets.back() + 4;
    BB.IsEntry = (BI == 0);

    for (size_t I = StartIdx; I < EndIdx; ++I)
      BB.InstIndices.push_back(I);

    if (!BB.InstIndices.empty()) {
      size_t LastIdx = BB.InstIndices.back();
      const MCInst &LastInst = Instructions[LastIdx];
      BB.HasTerminator = isBranch(LastInst) || isReturn(LastInst);
    }

    unsigned BlockIdx = CFG.Blocks.size();
    CFG.OffsetToBlockIndex[LeaderOffset] = BlockIdx;
    CFG.Blocks.push_back(BB);
  }

  // Third pass: connect edges
  for (unsigned BI = 0; BI < CFG.Blocks.size(); ++BI) {
    LiftedBasicBlock &BB = CFG.Blocks[BI];
    if (BB.InstIndices.empty())
      continue;

    size_t LastIdx = BB.InstIndices.back();
    const MCInst &LastInst = Instructions[LastIdx];

    if (isBranch(LastInst)) {
      int64_t Target = getBranchTargetOffset(LastInst, Offsets[LastIdx]);
      if (Target >= 0) {
        auto It = CFG.OffsetToBlockIndex.find(static_cast<uint64_t>(Target));
        if (It != CFG.OffsetToBlockIndex.end()) {
          BB.SuccessorIndices.push_back(It->second);
          CFG.Blocks[It->second].PredecessorIndices.push_back(BI);
        }
      }

      // Conditional branches fall through
      if (isConditionalBranch(LastInst) && BI + 1 < CFG.Blocks.size()) {
        BB.SuccessorIndices.push_back(BI + 1);
        CFG.Blocks[BI + 1].PredecessorIndices.push_back(BI);
      }
    } else if (!isReturn(LastInst)) {
      // Non-terminator: fall through
      if (BI + 1 < CFG.Blocks.size()) {
        BB.SuccessorIndices.push_back(BI + 1);
        CFG.Blocks[BI + 1].PredecessorIndices.push_back(BI);
      }
    }
  }

  Statistics.NumBasicBlocks = CFG.Blocks.size();

  LLVM_DEBUG(dbgs() << "Built CFG with " << CFG.Blocks.size() << " blocks\n");
}

MachineFunction &MIRLifter::createMachineFunction(Function &F) {
  return MMI->getOrCreateMachineFunction(F);
}

const TargetRegisterClass *MIRLifter::getRegClass(MCRegister Reg) const {
  // Note: This requires a function to get subtarget, but we don't have one here
  // For now, return nullptr - we'll fix this when we have proper context
  return nullptr;
}

MachineOperand MIRLifter::convertOperand(const MCOperand &Op,
                                         const MCInstrDesc &Desc,
                                         unsigned OpIdx,
                                         MachineFunction &MF) {
  if (Op.isReg()) {
    MCRegister Reg = Op.getReg();
    bool IsDef = OpIdx < Desc.getNumDefs();
    // For physical registers, we just use them directly
    return MachineOperand::CreateReg(Reg, IsDef, /*isImp=*/false,
                                     /*isKill=*/false, /*isDead=*/false);
  }

  if (Op.isImm()) {
    return MachineOperand::CreateImm(Op.getImm());
  }

  if (Op.isDFPImm()) {
    // For now, convert DFP immediate to regular immediate
    // Full handling would require ConstantFP creation
    return MachineOperand::CreateImm(static_cast<int64_t>(Op.getDFPImm()));
  }

  if (Op.isSFPImm()) {
    // For now, convert SFP immediate to regular immediate
    return MachineOperand::CreateImm(static_cast<int64_t>(Op.getSFPImm()));
  }

  // Default: create an undefined operand
  return MachineOperand::CreateReg(0, false);
}

Error MIRLifter::convertInstruction(const MCInst &Inst, MachineBasicBlock &MBB,
                                    MachineFunction &MF) {
  unsigned Opcode = Inst.getOpcode();
  const MCInstrDesc &Desc = MCII.get(Opcode);
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  // Get the corresponding MachineInstr opcode
  // For most instructions, the MCInst opcode should match
  const MCInstrDesc &MIDesc = TII->get(Opcode);

  MachineInstrBuilder MIB = BuildMI(MBB, MBB.end(), DebugLoc(), MIDesc);

  // Add operands
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    MachineOperand MO = convertOperand(Op, Desc, I, MF);
    MIB.add(MO);
  }

  // Add implicit operands from the instruction description
  // The MachineInstrBuilder should handle implicit defs/uses automatically

  return Error::success();
}

Expected<MachineFunction *>
MIRLifter::lift(ArrayRef<MCInst> Instructions, ArrayRef<uint64_t> Offsets,
                StringRef FunctionName) {
  if (Instructions.empty())
    return createStringError(inconvertibleErrorCode(),
                             "No instructions to lift");

  if (Instructions.size() != Offsets.size())
    return createStringError(inconvertibleErrorCode(),
                             "Instruction count doesn't match offset count");

  Statistics.NumInstructions = Instructions.size();

  // Build CFG
  buildCFG(Instructions, Offsets);

  if (Verbose) {
    errs() << "  Lifting " << Instructions.size() << " instructions in "
           << CFG.Blocks.size() << " basic blocks\n";
  }

  // Create a dummy LLVM IR function
  // For AMDGPU kernels, we use the amdgpu_kernel calling convention
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), false);
  Function *F = Function::Create(FT, GlobalValue::ExternalLinkage,
                                 FunctionName, TheModule.get());
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);

  // Add a dummy entry block to the IR function
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", F);
  new UnreachableInst(Ctx, Entry);

  // Create MachineFunction
  MachineFunction &MF = createMachineFunction(*F);
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Set that we're using physical registers only
  MRI.freezeReservedRegs();

  // Create MachineBasicBlocks
  BlockMap.clear();
  for (unsigned BI = 0; BI < CFG.Blocks.size(); ++BI) {
    MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
    MF.push_back(MBB);
    BlockMap[BI] = MBB;
  }

  // Connect successor/predecessor edges
  for (unsigned BI = 0; BI < CFG.Blocks.size(); ++BI) {
    const LiftedBasicBlock &BB = CFG.Blocks[BI];
    MachineBasicBlock *MBB = BlockMap[BI];

    for (unsigned SuccIdx : BB.SuccessorIndices) {
      MachineBasicBlock *SuccMBB = BlockMap[SuccIdx];
      if (SuccMBB)
        MBB->addSuccessor(SuccMBB);
    }
  }

  // Convert instructions
  for (unsigned BI = 0; BI < CFG.Blocks.size(); ++BI) {
    const LiftedBasicBlock &BB = CFG.Blocks[BI];
    MachineBasicBlock *MBB = BlockMap[BI];

    for (size_t InstIdx : BB.InstIndices) {
      const MCInst &Inst = Instructions[InstIdx];
      if (auto Err = convertInstruction(Inst, *MBB, MF)) {
        ++Statistics.NumUnknownInsts;
        LLVM_DEBUG(dbgs() << "Failed to convert instruction at index "
                          << InstIdx << "\n");
        // Continue anyway - we'll see what happens
      }
    }
  }

  if (Verbose) {
    errs() << "  Created MachineFunction with " << MF.size()
           << " basic blocks\n";
  }

  return &MF;
}
