//===-- LivenessAnalyzer.h - VGPR Liveness Analysis for MCInst --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the LivenessAnalyzer class which performs backward
// dataflow liveness analysis on MCInst streams for AMDGPU.
//
// The analysis is used to find dead registers at each program point, enabling
// safe register allocation for instruction emulation sequences.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_AMDGPU_RETARGET_LIVENESSANALYZER_H
#define LLVM_TOOLS_LLVM_AMDGPU_RETARGET_LIVENESSANALYZER_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include <vector>

namespace llvm {

/// Represents a basic block in the MCInst CFG.
struct MCBasicBlock {
  uint64_t StartOffset = 0;
  uint64_t EndOffset = 0;
  SmallVector<size_t, 8> InstIndices;       // Indices into instruction array
  SmallVector<unsigned, 4> Successors;       // Block indices
  SmallVector<unsigned, 4> Predecessors;     // Block indices
};

/// Control flow graph built from MCInst stream.
struct MCCFG {
  std::vector<MCBasicBlock> Blocks;
  DenseMap<uint64_t, unsigned> OffsetToBlock;  // Offset -> Block index
};

/// Register def/use information for an instruction.
struct RegDefUse {
  BitVector Defs;  // Registers defined by this instruction
  BitVector Uses;  // Registers used by this instruction

  RegDefUse(unsigned NumRegs) : Defs(NumRegs), Uses(NumRegs) {}
};

/// Liveness information at each program point.
struct LivenessInfo {
  std::vector<BitVector> LiveBefore;  // Live registers before each instruction
  std::vector<BitVector> LiveAfter;   // Live registers after each instruction
  bool Converged = false;
};

/// Performs backward dataflow liveness analysis on MCInst streams.
///
/// This class analyzes an AMDGPU MCInst stream to determine which VGPR
/// registers are live at each program point. This information is used to
/// find dead registers that can be safely used as scratch for instruction
/// emulation sequences.
///
/// The analysis:
/// 1. Builds a CFG from the instruction stream (detecting branches)
/// 2. Extracts def/use information for each instruction
/// 3. Runs backward dataflow iteration until convergence
/// 4. Provides queries for dead registers at any program point
class LivenessAnalyzer {
public:
  LivenessAnalyzer(const MCInstrInfo &MCII, const MCRegisterInfo &MRI);

  /// Analyze the given instruction stream.
  /// Returns true if analysis succeeded.
  bool analyze(ArrayRef<MCInst> Instructions, ArrayRef<uint64_t> Offsets);

  /// Get the set of dead VGPR registers at the given instruction index.
  /// Returns a BitVector where set bits indicate dead registers.
  BitVector getDeadVGPRsAt(size_t InstIndex) const;

  /// Allocate a scratch VGPR that is dead at the given instruction.
  /// Returns the VGPR number (0-255), or -1 if no register is available.
  /// PreferAbove: prefer registers >= this value (to avoid low registers)
  int allocateScratchVGPR(size_t InstIndex, unsigned PreferAbove = 0) const;

  /// Allocate a 64-bit scratch VGPR pair (must be consecutive even/odd pair).
  /// Returns the low register number, or -1 if not available.
  int allocateScratchVGPR64(size_t InstIndex, unsigned PreferAbove = 0) const;

  /// Get statistics about the analysis.
  struct Stats {
    unsigned NumInstructions = 0;
    unsigned NumBasicBlocks = 0;
    unsigned NumIterations = 0;
    unsigned MaxLiveVGPRs = 0;
    unsigned MinDeadVGPRs = 0;
  };
  const Stats &getStats() const { return Statistics; }

private:
  /// Build CFG from instruction stream.
  void buildCFG(ArrayRef<MCInst> Instructions, ArrayRef<uint64_t> Offsets);

  /// Extract register def/use information for an instruction.
  RegDefUse getRegDefUse(const MCInst &Inst) const;

  /// Check if instruction is a branch (conditional or unconditional).
  bool isBranch(const MCInst &Inst) const;

  /// Check if instruction is a conditional branch.
  bool isConditionalBranch(const MCInst &Inst) const;

  /// Get branch target offset from instruction.
  int64_t getBranchOffset(const MCInst &Inst, uint64_t InstOffset) const;

  /// Run backward dataflow iteration.
  void runDataflow();

  /// Get VGPR number from register (0-255), or -1 if not a VGPR.
  int getVGPRNumber(unsigned Reg) const;

  /// Check if register is a VGPR or part of a VGPR.
  bool isVGPR(unsigned Reg) const;

  /// Get all VGPR sub-registers affected by this register.
  void getVGPRsForReg(unsigned Reg, BitVector &VGPRs) const;

  const MCInstrInfo &MCII;
  const MCRegisterInfo &MRI;

  // Analysis results
  MCCFG CFG;
  std::vector<RegDefUse> DefUses;
  LivenessInfo Liveness;
  Stats Statistics;

  // Number of VGPR registers (256 for AMDGPU)
  static constexpr unsigned NumVGPRs = 256;
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_AMDGPU_RETARGET_LIVENESSANALYZER_H
