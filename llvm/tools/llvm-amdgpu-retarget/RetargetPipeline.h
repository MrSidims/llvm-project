//===-- RetargetPipeline.h - Full MCInst->MIR->Code Pipeline ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the RetargetPipeline class which implements the full
// MCInst -> MachineFunction -> Target Code pipeline using LLVM's backend.
//
// The key insight is that we can:
// 1. Lift MCInst to MachineFunction (with physical registers from source arch)
// 2. Transform instructions for target arch (opcode mapping, expansion)
// 3. Use LLVM's TargetPassConfig to run necessary passes
// 4. Emit code using the standard emission pipeline
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_AMDGPU_RETARGET_RETARGETPIPELINE_H
#define LLVM_TOOLS_LLVM_AMDGPU_RETARGET_RETARGETPIPELINE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"
#include <memory>

namespace llvm {

class LLVMContext;
class MachineFunction;
class MCDisassembler;
class MCInstrInfo;
class MCRegisterInfo;
class MCSubtargetInfo;
class MemoryBuffer;
class MIRLifter;
class Module;
class Target;
class TargetMachine;

/// The complete retargeting pipeline.
///
/// This class orchestrates the full binary retargeting process:
///
/// 1. **Disassembly**: Parse ELF, disassemble .text to MCInst stream
/// 2. **Lifting**: Convert MCInst stream to MachineFunction
/// 3. **Transformation**: Adapt instructions for target architecture
///    - Simple opcode mapping for compatible instructions
///    - Instruction expansion for unsupported instructions
///    - Register re-allocation where needed
/// 4. **Code Generation**: Run LLVM backend passes and emit code
/// 5. **ELF Writing**: Create output code object with updated metadata
///
/// The pipeline uses LLVM's standard infrastructure:
/// - TargetMachine for target-specific information
/// - MachineFunction/MachineBasicBlock/MachineInstr representation
/// - TargetPassConfig for running optimization passes
/// - MCStreamer for code emission
class RetargetPipeline {
public:
  /// Create a retargeting pipeline.
  ///
  /// \param SourceCPU Source GPU architecture (e.g., "gfx942")
  /// \param TargetCPU Target GPU architecture (e.g., "gfx90a")
  /// \param Verbose Enable verbose output
  RetargetPipeline(StringRef SourceCPU, StringRef TargetCPU, bool Verbose = false);

  ~RetargetPipeline();

  /// Initialize the pipeline (create TargetMachines, etc.)
  Error initialize();

  /// Run the full retargeting pipeline on an input file.
  ///
  /// \param InputBuffer The input ELF code object.
  /// \param OutputPath Path for the output file.
  /// \returns Error on failure.
  Error run(const MemoryBuffer &InputBuffer, StringRef OutputPath);

  /// Get statistics about the pipeline execution.
  struct Stats {
    unsigned NumInstructions = 0;
    unsigned NumBasicBlocks = 0;
    unsigned NumTransformed = 0;
    unsigned NumExpanded = 0;
    unsigned InputBytes = 0;
    unsigned OutputBytes = 0;
  };
  const Stats &getStats() const { return Statistics; }

private:
  /// Set up target machines for source and target architectures.
  Error setupTargets();

  /// Disassemble .text section to MCInst stream.
  Error disassemble(ArrayRef<uint8_t> TextSection,
                    SmallVectorImpl<MCInst> &Instructions,
                    SmallVectorImpl<uint64_t> &Offsets);

  /// Lift MCInst stream to MachineFunction.
  Error liftToMIR(ArrayRef<MCInst> Instructions, ArrayRef<uint64_t> Offsets,
                  StringRef FunctionName);

  /// Transform MachineFunction for target architecture.
  Error transformForTarget();

  /// Emit code from MachineFunction.
  Error emitCode(SmallVectorImpl<char> &Output);

  std::string SourceCPU;
  std::string TargetCPU;
  bool Verbose;

  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> TheModule;

  // Target infrastructure
  const Target *TheTarget = nullptr;
  Triple TheTriple;
  std::unique_ptr<TargetMachine> SourceTM;
  std::unique_ptr<TargetMachine> TargetTM;

  // MC components for disassembly
  std::unique_ptr<MCInstrInfo> SourceMCII;
  std::unique_ptr<MCRegisterInfo> SourceMRI;
  std::unique_ptr<MCSubtargetInfo> SourceSTI;
  std::unique_ptr<MCDisassembler> SourceDisasm;

  // MIR Lifter - created during processing
  std::unique_ptr<MIRLifter> Lifter;
  MachineFunction *CurrentMF = nullptr;

  Stats Statistics;
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_AMDGPU_RETARGET_RETARGETPIPELINE_H
