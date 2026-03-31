//===-- LivenessAnalyzer.cpp - VGPR Liveness Analysis for MCInst ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LivenessAnalyzer.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-liveness"

LivenessAnalyzer::LivenessAnalyzer(const MCInstrInfo &MCII,
                                   const MCRegisterInfo &MRI)
    : MCII(MCII), MRI(MRI) {}

bool LivenessAnalyzer::analyze(ArrayRef<MCInst> Instructions,
                               ArrayRef<uint64_t> Offsets) {
  if (Instructions.empty())
    return true;

  Statistics.NumInstructions = Instructions.size();

  // Build CFG
  buildCFG(Instructions, Offsets);
  Statistics.NumBasicBlocks = CFG.Blocks.size();

  // Extract def/use for each instruction
  DefUses.clear();
  DefUses.reserve(Instructions.size());
  for (const MCInst &Inst : Instructions) {
    DefUses.push_back(getRegDefUse(Inst));
  }

  // Initialize liveness info
  Liveness.LiveBefore.assign(Instructions.size(), BitVector(NumVGPRs));
  Liveness.LiveAfter.assign(Instructions.size(), BitVector(NumVGPRs));

  // Run backward dataflow
  runDataflow();

  // Compute statistics
  unsigned MaxLive = 0;
  unsigned MinDead = NumVGPRs;
  for (size_t I = 0; I < Instructions.size(); ++I) {
    unsigned NumLive = Liveness.LiveBefore[I].count();
    unsigned NumDead = NumVGPRs - NumLive;
    MaxLive = std::max(MaxLive, NumLive);
    MinDead = std::min(MinDead, NumDead);
  }
  Statistics.MaxLiveVGPRs = MaxLive;
  Statistics.MinDeadVGPRs = MinDead;

  LLVM_DEBUG(dbgs() << "Liveness analysis complete:\n"
                    << "  Instructions: " << Statistics.NumInstructions << "\n"
                    << "  Basic blocks: " << Statistics.NumBasicBlocks << "\n"
                    << "  Iterations: " << Statistics.NumIterations << "\n"
                    << "  Max live VGPRs: " << Statistics.MaxLiveVGPRs << "\n"
                    << "  Min dead VGPRs: " << Statistics.MinDeadVGPRs << "\n");

  return true;
}

void LivenessAnalyzer::buildCFG(ArrayRef<MCInst> Instructions,
                                ArrayRef<uint64_t> Offsets) {
  CFG.Blocks.clear();
  CFG.OffsetToBlock.clear();

  if (Instructions.empty())
    return;

  // First pass: identify basic block boundaries (branch targets and fall-through)
  std::set<uint64_t> Leaders;
  Leaders.insert(Offsets[0]); // First instruction is a leader

  for (size_t I = 0; I < Instructions.size(); ++I) {
    const MCInst &Inst = Instructions[I];
    if (isBranch(Inst)) {
      int64_t TargetOffset = getBranchOffset(Inst, Offsets[I]);
      if (TargetOffset >= 0)
        Leaders.insert(static_cast<uint64_t>(TargetOffset));

      // Next instruction (if any) is also a leader
      if (I + 1 < Instructions.size())
        Leaders.insert(Offsets[I + 1]);
    }
  }

  // Build offset to instruction index map
  DenseMap<uint64_t, size_t> OffsetToInst;
  for (size_t I = 0; I < Offsets.size(); ++I)
    OffsetToInst[Offsets[I]] = I;

  // Second pass: create basic blocks
  std::vector<uint64_t> SortedLeaders(Leaders.begin(), Leaders.end());
  std::sort(SortedLeaders.begin(), SortedLeaders.end());

  for (size_t BI = 0; BI < SortedLeaders.size(); ++BI) {
    uint64_t LeaderOffset = SortedLeaders[BI];

    // Find the instruction index for this leader
    auto It = OffsetToInst.find(LeaderOffset);
    if (It == OffsetToInst.end())
      continue; // Leader offset not in our instruction stream

    size_t StartIdx = It->second;

    // Find end of this block
    size_t EndIdx = Instructions.size();
    if (BI + 1 < SortedLeaders.size()) {
      auto EndIt = OffsetToInst.find(SortedLeaders[BI + 1]);
      if (EndIt != OffsetToInst.end())
        EndIdx = EndIt->second;
    }

    MCBasicBlock BB;
    BB.StartOffset = LeaderOffset;
    BB.EndOffset = (EndIdx < Offsets.size()) ? Offsets[EndIdx] : Offsets.back() + 4;

    for (size_t I = StartIdx; I < EndIdx; ++I)
      BB.InstIndices.push_back(I);

    unsigned BlockIdx = CFG.Blocks.size();
    CFG.OffsetToBlock[LeaderOffset] = BlockIdx;
    CFG.Blocks.push_back(BB);
  }

  // Third pass: connect edges
  for (unsigned BI = 0; BI < CFG.Blocks.size(); ++BI) {
    MCBasicBlock &BB = CFG.Blocks[BI];
    if (BB.InstIndices.empty())
      continue;

    size_t LastIdx = BB.InstIndices.back();
    const MCInst &LastInst = Instructions[LastIdx];

    if (isBranch(LastInst)) {
      int64_t TargetOffset = getBranchOffset(LastInst, Offsets[LastIdx]);
      if (TargetOffset >= 0) {
        auto It = CFG.OffsetToBlock.find(static_cast<uint64_t>(TargetOffset));
        if (It != CFG.OffsetToBlock.end()) {
          BB.Successors.push_back(It->second);
          CFG.Blocks[It->second].Predecessors.push_back(BI);
        }
      }

      // Conditional branches also fall through
      if (isConditionalBranch(LastInst) && BI + 1 < CFG.Blocks.size()) {
        BB.Successors.push_back(BI + 1);
        CFG.Blocks[BI + 1].Predecessors.push_back(BI);
      }
    } else {
      // Non-branch: fall through to next block
      if (BI + 1 < CFG.Blocks.size()) {
        BB.Successors.push_back(BI + 1);
        CFG.Blocks[BI + 1].Predecessors.push_back(BI);
      }
    }
  }

  LLVM_DEBUG(dbgs() << "Built CFG with " << CFG.Blocks.size() << " blocks\n");
}

RegDefUse LivenessAnalyzer::getRegDefUse(const MCInst &Inst) const {
  RegDefUse DU(NumVGPRs);

  const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());

  // Process explicit operands
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (!Op.isReg())
      continue;

    unsigned Reg = Op.getReg();
    if (!isVGPR(Reg))
      continue;

    // Check if this operand is a def or use based on MCInstrDesc
    bool IsDef = false;
    if (I < Desc.getNumOperands()) {
      const MCOperandInfo &OpInfo = Desc.operands()[I];
      IsDef = (OpInfo.isOptionalDef() ||
               (I < Desc.getNumDefs()));
    }

    BitVector AffectedVGPRs(NumVGPRs);
    getVGPRsForReg(Reg, AffectedVGPRs);

    if (IsDef)
      DU.Defs |= AffectedVGPRs;
    else
      DU.Uses |= AffectedVGPRs;
  }

  // Process implicit defs/uses from MCInstrDesc
  for (MCPhysReg R : Desc.implicit_defs()) {
    if (isVGPR(R)) {
      BitVector AffectedVGPRs(NumVGPRs);
      getVGPRsForReg(R, AffectedVGPRs);
      DU.Defs |= AffectedVGPRs;
    }
  }

  for (MCPhysReg R : Desc.implicit_uses()) {
    if (isVGPR(R)) {
      BitVector AffectedVGPRs(NumVGPRs);
      getVGPRsForReg(R, AffectedVGPRs);
      DU.Uses |= AffectedVGPRs;
    }
  }

  return DU;
}

bool LivenessAnalyzer::isBranch(const MCInst &Inst) const {
  const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
  return Desc.isBranch() || Desc.isCall() || Desc.isReturn();
}

bool LivenessAnalyzer::isConditionalBranch(const MCInst &Inst) const {
  const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
  return Desc.isConditionalBranch();
}

int64_t LivenessAnalyzer::getBranchOffset(const MCInst &Inst,
                                          uint64_t InstOffset) const {
  // AMDGPU branch instructions have a PC-relative offset in an immediate operand
  // The format is: target = PC + 4 + (simm16 * 4)
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isImm()) {
      int64_t Simm16 = Op.getImm();
      // Check if this looks like a branch offset (signed 16-bit range * 4)
      if (Simm16 >= -32768 && Simm16 <= 32767) {
        return static_cast<int64_t>(InstOffset) + 4 + (Simm16 * 4);
      }
    }
  }
  return -1; // Unknown target
}

void LivenessAnalyzer::runDataflow() {
  // Backward dataflow iteration
  // LiveBefore[I] = Uses[I] ∪ (LiveAfter[I] - Defs[I])
  // LiveAfter[I] = ∪{LiveBefore[S] : S is successor of I}

  bool Changed = true;
  unsigned Iterations = 0;
  const unsigned MaxIterations = 100;

  while (Changed && Iterations < MaxIterations) {
    Changed = false;
    ++Iterations;

    // Process blocks in reverse order (approximating reverse post-order)
    for (int BI = CFG.Blocks.size() - 1; BI >= 0; --BI) {
      const MCBasicBlock &BB = CFG.Blocks[BI];

      // Compute LiveOut for this block (union of LiveIn of successors)
      BitVector BlockLiveOut(NumVGPRs);
      for (unsigned SuccIdx : BB.Successors) {
        if (!CFG.Blocks[SuccIdx].InstIndices.empty()) {
          size_t SuccFirstInst = CFG.Blocks[SuccIdx].InstIndices.front();
          BlockLiveOut |= Liveness.LiveBefore[SuccFirstInst];
        }
      }

      // Process instructions in reverse order within block
      BitVector Live = BlockLiveOut;

      for (int II = BB.InstIndices.size() - 1; II >= 0; --II) {
        size_t InstIdx = BB.InstIndices[II];

        // LiveAfter[I] = Live (from successors or next instruction)
        BitVector OldLiveAfter = Liveness.LiveAfter[InstIdx];
        Liveness.LiveAfter[InstIdx] = Live;
        if (OldLiveAfter != Live)
          Changed = true;

        // LiveBefore[I] = Uses[I] ∪ (LiveAfter[I] - Defs[I])
        BitVector LiveBefore = Live;
        LiveBefore.reset(DefUses[InstIdx].Defs);
        LiveBefore |= DefUses[InstIdx].Uses;

        BitVector OldLiveBefore = Liveness.LiveBefore[InstIdx];
        Liveness.LiveBefore[InstIdx] = LiveBefore;
        if (OldLiveBefore != LiveBefore)
          Changed = true;

        Live = LiveBefore;
      }
    }
  }

  Statistics.NumIterations = Iterations;
  Liveness.Converged = !Changed;

  if (!Liveness.Converged) {
    LLVM_DEBUG(dbgs() << "Warning: Liveness analysis did not converge after "
                      << MaxIterations << " iterations\n");
  }
}

int LivenessAnalyzer::getVGPRNumber(unsigned Reg) const {
  // AMDGPU VGPR_32 registers are named VGPR0, VGPR1, ..., VGPR255
  StringRef Name = MRI.getName(Reg);
  if (Name.consume_front("VGPR")) {
    int N;
    if (!Name.getAsInteger(10, N) && N >= 0 && N < 256)
      return N;
  }
  return -1;
}

bool LivenessAnalyzer::isVGPR(unsigned Reg) const {
  // Check if this register or any of its super-registers is a VGPR
  StringRef Name = MRI.getName(Reg);

  // Direct VGPR
  if (Name.starts_with("VGPR"))
    return true;

  // Check super-registers
  for (MCPhysReg SR : MRI.superregs(Reg)) {
    StringRef SRName = MRI.getName(SR);
    if (SRName.starts_with("VGPR"))
      return true;
  }

  return false;
}

void LivenessAnalyzer::getVGPRsForReg(unsigned Reg, BitVector &VGPRs) const {
  // Get the VGPR number for this register
  int N = getVGPRNumber(Reg);
  if (N >= 0) {
    VGPRs.set(N);
    return;
  }

  // For register tuples (e.g., VGPR0_VGPR1), extract component VGPRs
  StringRef Name = MRI.getName(Reg);
  if (Name.contains("VGPR")) {
    // Try to parse VGPR numbers from the name
    // Format: VGPRx_VGPRy or VGPR[x:y]
    size_t Pos = 0;
    while (Pos < Name.size()) {
      size_t Start = Name.find("VGPR", Pos);
      if (Start == StringRef::npos)
        break;

      Start += 4; // Skip "VGPR"
      size_t End = Start;
      while (End < Name.size() && isdigit(Name[End]))
        ++End;

      if (End > Start) {
        int Num;
        if (!Name.substr(Start, End - Start).getAsInteger(10, Num)) {
          if (Num >= 0 && Num < 256)
            VGPRs.set(Num);
        }
      }
      Pos = End;
    }
  }

  // Also check sub-registers
  for (MCPhysReg SR : MRI.subregs(Reg)) {
    int SubN = getVGPRNumber(SR);
    if (SubN >= 0)
      VGPRs.set(SubN);
  }
}

BitVector LivenessAnalyzer::getDeadVGPRsAt(size_t InstIndex) const {
  if (InstIndex >= Liveness.LiveBefore.size())
    return BitVector(NumVGPRs, true); // All dead if out of bounds

  BitVector Dead(NumVGPRs, true);
  Dead.reset(Liveness.LiveBefore[InstIndex]);
  return Dead;
}

int LivenessAnalyzer::allocateScratchVGPR(size_t InstIndex,
                                          unsigned PreferAbove) const {
  BitVector Dead = getDeadVGPRsAt(InstIndex);

  // First try to find a register >= PreferAbove
  for (unsigned I = PreferAbove; I < NumVGPRs; ++I) {
    if (Dead.test(I))
      return static_cast<int>(I);
  }

  // Fall back to any dead register
  for (unsigned I = 0; I < PreferAbove; ++I) {
    if (Dead.test(I))
      return static_cast<int>(I);
  }

  return -1; // No dead register available
}

int LivenessAnalyzer::allocateScratchVGPR64(size_t InstIndex,
                                            unsigned PreferAbove) const {
  BitVector Dead = getDeadVGPRsAt(InstIndex);

  // 64-bit pairs must be even-aligned: (v0,v1), (v2,v3), etc.
  // Start from PreferAbove, rounded up to even
  unsigned Start = (PreferAbove + 1) & ~1u;

  for (unsigned I = Start; I + 1 < NumVGPRs; I += 2) {
    if (Dead.test(I) && Dead.test(I + 1))
      return static_cast<int>(I);
  }

  // Fall back to lower registers
  for (unsigned I = 0; I + 1 < Start && I + 1 < NumVGPRs; I += 2) {
    if (Dead.test(I) && Dead.test(I + 1))
      return static_cast<int>(I);
  }

  return -1; // No dead pair available
}
