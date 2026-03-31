//===-- MIRLifter.h - Lift MCInst to MachineFunction -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MIRLifter class which lifts an MCInst stream to a
// MachineFunction for recompilation through the LLVM backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_AMDGPU_RETARGET_MIRLIFTER_H
#define LLVM_TOOLS_LLVM_AMDGPU_RETARGET_MIRLIFTER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>

namespace llvm {

class MCSubtargetInfo;

/// Basic block information extracted from MCInst stream.
struct LiftedBasicBlock {
  uint64_t StartOffset = 0;
  uint64_t EndOffset = 0;
  SmallVector<size_t, 64> InstIndices;
  SmallVector<unsigned, 4> SuccessorIndices;
  SmallVector<unsigned, 4> PredecessorIndices;
  bool IsEntry = false;
  bool HasTerminator = false;
};

/// Control flow graph extracted from MCInst stream.
struct LiftedCFG {
  std::vector<LiftedBasicBlock> Blocks;
  DenseMap<uint64_t, unsigned> OffsetToBlockIndex;
};

/// Lifts an MCInst stream to a MachineFunction.
///
/// This class performs the following:
/// 1. Builds a CFG from the instruction stream (detecting branches)
/// 2. Creates a MachineFunction with MachineBasicBlocks
/// 3. Converts each MCInst to a MachineInstr using physical registers
/// 4. Reconstructs basic liveness information
///
/// The resulting MachineFunction can be passed through LLVM's backend
/// passes for optimization and code emission.
class MIRLifter {
public:
  MIRLifter(LLVMContext &Ctx, const TargetMachine &TM, const MCInstrInfo &MCII,
            const MCRegisterInfo &MRI, bool Verbose = false);

  ~MIRLifter();

  /// Lift an MCInst stream to a MachineFunction.
  ///
  /// \param Instructions The disassembled MCInst stream.
  /// \param Offsets The byte offset of each instruction.
  /// \param FunctionName Name for the generated function.
  /// \returns The lifted MachineFunction, or an error.
  Expected<MachineFunction *> lift(ArrayRef<MCInst> Instructions,
                                   ArrayRef<uint64_t> Offsets,
                                   StringRef FunctionName);

  /// Get the Module containing the lifted functions.
  Module &getModule() { return *TheModule; }

  /// Get the MachineModuleInfo.
  MachineModuleInfo &getMMI() { return *MMI; }

  /// Get statistics about the lifting process.
  struct Stats {
    unsigned NumInstructions = 0;
    unsigned NumBasicBlocks = 0;
    unsigned NumBranches = 0;
    unsigned NumUnknownInsts = 0;
  };
  const Stats &getStats() const { return Statistics; }

private:
  /// Build CFG from instruction stream.
  void buildCFG(ArrayRef<MCInst> Instructions, ArrayRef<uint64_t> Offsets);

  /// Check if instruction is a branch.
  bool isBranch(const MCInst &Inst) const;

  /// Check if instruction is a conditional branch.
  bool isConditionalBranch(const MCInst &Inst) const;

  /// Check if instruction is a return/end program.
  bool isReturn(const MCInst &Inst) const;

  /// Get branch target offset.
  int64_t getBranchTargetOffset(const MCInst &Inst, uint64_t InstOffset) const;

  /// Create the Module and MachineModuleInfo.
  void initializeModule();

  /// Create a MachineFunction for the given function.
  MachineFunction &createMachineFunction(Function &F);

  /// Convert MCInst to MachineInstr and add to MBB.
  Error convertInstruction(const MCInst &Inst, MachineBasicBlock &MBB,
                           MachineFunction &MF);

  /// Map MCOperand to MachineOperand.
  MachineOperand convertOperand(const MCOperand &Op, const MCInstrDesc &Desc,
                                unsigned OpIdx, MachineFunction &MF);

  /// Get the register class for a physical register.
  const TargetRegisterClass *getRegClass(MCRegister Reg) const;

  LLVMContext &Ctx;
  const TargetMachine &TM;
  const MCInstrInfo &MCII;
  const MCRegisterInfo &MRI;
  bool Verbose;

  std::unique_ptr<Module> TheModule;
  std::unique_ptr<MachineModuleInfo> MMI;

  LiftedCFG CFG;
  Stats Statistics;

  // Map from block index to MachineBasicBlock
  DenseMap<unsigned, MachineBasicBlock *> BlockMap;
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_AMDGPU_RETARGET_MIRLIFTER_H
