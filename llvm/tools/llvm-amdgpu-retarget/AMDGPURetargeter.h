//===-- AMDGPURetargeter.h - AMDGPU Instruction Retargeting -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AMDGPURetargeter class which transforms instructions
// from one AMDGPU architecture to another.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_AMDGPU_RETARGET_AMDGPURETARGETER_H
#define LLVM_TOOLS_LLVM_AMDGPU_RETARGET_AMDGPURETARGETER_H

#include "LivenessAnalyzer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace llvm {

/// Retargets AMDGPU instructions from one architecture to another.
///
/// This class handles three types of transformations:
/// 1. Direct opcode mapping (1:1 instruction swaps)
/// 2. Emulation sequences (1:N expansion for unsupported instructions)
/// 3. Pass-through (instructions that work on both architectures)
class AMDGPURetargeter {
public:
  AMDGPURetargeter(StringRef SourceCPU, StringRef TargetCPU,
                   const MCInstrInfo &SourceMCII, const MCInstrInfo &TargetMCII,
                   const MCRegisterInfo &SourceMRI,
                   const MCRegisterInfo &TargetMRI);

  /// Analyze the entire instruction stream for liveness information.
  /// This must be called before transformAll() if register allocation is needed.
  Error analyzeForLiveness(ArrayRef<MCInst> Instructions,
                           ArrayRef<uint64_t> Offsets);

  /// Transform all instructions in the stream.
  /// Uses liveness information (if available) for optimal register allocation.
  Error transformAll(ArrayRef<MCInst> SourceInsts, ArrayRef<uint64_t> Offsets,
                     SmallVectorImpl<MCInst> &TargetInsts);

  /// Transform a single source instruction into one or more target instructions.
  /// Uses the current instruction index for liveness-based register allocation.
  /// Returns an error if the instruction cannot be retargeted.
  Error transform(const MCInst &SourceInst, size_t InstIndex,
                  SmallVectorImpl<MCInst> &TargetInsts);

  /// Check if a source instruction requires emulation on the target.
  bool requiresEmulation(unsigned SourceOpcode) const;

  /// Get statistics about the transformation.
  struct Stats {
    unsigned PassThrough = 0;
    unsigned DirectMapped = 0;
    unsigned Emulated = 0;
    unsigned Unsupported = 0;
    unsigned ScratchRegsUsed = 0;      // Number of scratch registers allocated
    unsigned MaxExtraVGPRs = 0;        // Max extra VGPRs needed above kernel's declared count
  };
  const Stats &getStats() const { return Statistics; }

  /// Get liveness analysis statistics (if analyze() was called).
  const LivenessAnalyzer::Stats *getLivenessStats() const {
    return Liveness ? &Liveness->getStats() : nullptr;
  }

  /// Get the maximum number of extra VGPRs needed by this transformation.
  /// This should be added to the kernel's VGPR count in the kernel descriptor.
  unsigned getExtraVGPRsNeeded() const { return Statistics.MaxExtraVGPRs; }

private:
  /// Initialize opcode mapping tables for the source/target pair.
  void initializeOpcodeMap();

  /// Check if source is gfx950 and target is gfx942 (same-family retargeting).
  bool isGFX950ToGFX942() const;

  /// Check if source is gfx942 and target is gfx90a (cross-gen retargeting).
  bool isGFX942ToGFX90a() const;

  /// Check if both source and target are in the GFX9 family.
  bool isSameGFX9Family() const;

  /// Get the target opcode for a direct 1:1 mapping.
  /// Returns 0 if no direct mapping exists.
  unsigned getDirectMapping(unsigned SourceOpcode) const;

  /// Emit emulation sequence for an instruction that requires it.
  Error emitEmulationSequence(const MCInst &SourceInst,
                              SmallVectorImpl<MCInst> &TargetInsts);

  /// Emit bf16 pack emulation (v_cvt_pk_bf16_f32)
  Error emitBF16PackEmulation(const MCInst &SourceInst,
                              SmallVectorImpl<MCInst> &TargetInsts);

  /// Emit FP4 quantization emulation (v_cvt_scalef32_pk_fp4_f32)
  Error emitFP4QuantEmulation(const MCInst &SourceInst,
                              SmallVectorImpl<MCInst> &TargetInsts);

  /// Emit v_lshl_add_u64 emulation for gfx942 -> gfx90a
  /// Semantics: D.u64 = (S0.u64 << S1.u[2:0]) + S2.u64
  /// Emulation: v_lshlrev_b64 tmp, src1, src0 + v_add_u64 dst, tmp, src2
  Error emitLshlAddU64Emulation(const MCInst &SourceInst, size_t InstIndex,
                                SmallVectorImpl<MCInst> &TargetInsts);

  /// Allocate a scratch VGPR using liveness info, or fall back to v255.
  /// Returns the VGPR number (0-255).
  int allocateScratchVGPR(size_t InstIndex);

  /// Allocate a 64-bit scratch VGPR pair using liveness info.
  /// Returns the low register number, or -1 if not available.
  int allocateScratchVGPR64(size_t InstIndex);

  /// Get the MCRegister for a VGPR number.
  unsigned getVGPRRegister(unsigned VGPRNum) const;

  /// Get the MCRegister for a 64-bit VGPR pair.
  unsigned getVGPR64Register(unsigned LowVGPRNum) const;

  std::string SourceCPU;
  std::string TargetCPU;
  const MCInstrInfo &SourceMCII;
  const MCInstrInfo &TargetMCII;
  const MCRegisterInfo &SourceMRI;
  const MCRegisterInfo &TargetMRI;

  /// Direct opcode mapping from source to target.
  DenseMap<unsigned, unsigned> OpcodeMap;

  /// Set of opcodes that require emulation.
  DenseMap<unsigned, bool> EmulationRequired;

  /// Liveness analyzer (created by analyzeForLiveness).
  std::unique_ptr<LivenessAnalyzer> Liveness;

  /// Current instruction index (for liveness-aware allocation).
  size_t CurrentInstIndex = 0;

  /// Track the kernel's declared VGPR count (from kernel descriptor).
  unsigned KernelVGPRCount = 0;

  Stats Statistics;
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_AMDGPU_RETARGET_AMDGPURETARGETER_H
