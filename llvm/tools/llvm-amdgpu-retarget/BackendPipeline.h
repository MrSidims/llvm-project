//===-- BackendPipeline.h - Run LLVM Backend on MachineFunction -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the BackendPipeline class which runs LLVM backend passes
// on a MachineFunction to generate code for a target architecture.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_AMDGPU_RETARGET_BACKENDPIPELINE_H
#define LLVM_TOOLS_LLVM_AMDGPU_RETARGET_BACKENDPIPELINE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include <memory>

namespace llvm {

class Module;
class raw_pwrite_stream;
class MCContext;

/// Runs LLVM backend passes on a MachineFunction.
///
/// This class provides functionality to:
/// 1. Run post-RA passes on a MachineFunction with physical registers
/// 2. Perform instruction selection for the target architecture
/// 3. Emit machine code to a buffer or file
///
/// For retargeting, we:
/// 1. Take a MachineFunction created from lifted MCInst (source arch)
/// 2. Transform it for the target architecture
/// 3. Run backend passes (scheduling, code emission)
/// 4. Emit the final machine code
class BackendPipeline {
public:
  BackendPipeline(TargetMachine &SourceTM, TargetMachine &TargetTM,
                  bool Verbose = false);

  ~BackendPipeline();

  /// Transform a MachineFunction from source to target architecture.
  ///
  /// This handles instruction transformation:
  /// - Opcode mapping for compatible instructions
  /// - Instruction expansion for unsupported instructions
  ///
  /// \param MF The MachineFunction to transform (modified in place).
  /// \returns Error if transformation fails.
  Error transformFunction(MachineFunction &MF);

  /// Emit code for a MachineFunction.
  ///
  /// \param MF The MachineFunction to emit.
  /// \param MMI The MachineModuleInfo.
  /// \param OS Output stream for the generated code.
  /// \returns Error if emission fails.
  Error emitCode(MachineFunction &MF, MachineModuleInfo &MMI,
                 raw_pwrite_stream &OS);

  /// Run the full pipeline: transform + emit.
  ///
  /// \param MF The source MachineFunction.
  /// \param MMI The MachineModuleInfo.
  /// \param Output Buffer to receive the generated code.
  /// \returns Error if any step fails.
  Error run(MachineFunction &MF, MachineModuleInfo &MMI,
            SmallVectorImpl<char> &Output);

  /// Get statistics about the pipeline.
  struct Stats {
    unsigned InstructionsTransformed = 0;
    unsigned InstructionsExpanded = 0;
    unsigned BytesEmitted = 0;
  };
  const Stats &getStats() const { return Statistics; }

private:
  /// Check if an instruction needs transformation for the target.
  bool needsTransformation(const MachineInstr &MI) const;

  /// Check if an instruction needs expansion (1:N).
  bool needsExpansion(const MachineInstr &MI) const;

  /// Transform a single instruction.
  Error transformInstruction(MachineInstr &MI, MachineBasicBlock &MBB);

  /// Expand an instruction into a sequence.
  Error expandInstruction(MachineInstr &MI, MachineBasicBlock &MBB);

  /// Run post-RA scheduling and other cleanup passes.
  Error runPostRAPasses(MachineFunction &MF);

  TargetMachine &SourceTM;
  TargetMachine &TargetTM;
  bool Verbose;

  Stats Statistics;
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_AMDGPU_RETARGET_BACKENDPIPELINE_H
