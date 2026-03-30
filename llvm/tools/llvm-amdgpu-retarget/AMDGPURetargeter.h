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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Error.h"

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

  /// Transform a single source instruction into one or more target instructions.
  /// Returns an error if the instruction cannot be retargeted.
  Error transform(const MCInst &SourceInst,
                  SmallVectorImpl<MCInst> &TargetInsts);

  /// Check if a source instruction requires emulation on the target.
  bool requiresEmulation(unsigned SourceOpcode) const;

  /// Get statistics about the transformation.
  struct Stats {
    unsigned PassThrough = 0;
    unsigned DirectMapped = 0;
    unsigned Emulated = 0;
    unsigned Unsupported = 0;
  };
  const Stats &getStats() const { return Statistics; }

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
  Error emitLshlAddU64Emulation(const MCInst &SourceInst,
                                SmallVectorImpl<MCInst> &TargetInsts);

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

  Stats Statistics;
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_AMDGPU_RETARGET_AMDGPURETARGETER_H
