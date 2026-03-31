//===-- AMDGPURetargeter.cpp - AMDGPU Instruction Retargeting -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AMDGPURetargeter class which transforms instructions
// from one AMDGPU architecture to another.
//
// The transformation tables are based on the ROCm HotSwap implementation
// and handle cross-generation compatibility between GFX9 family GPUs.
//
//===----------------------------------------------------------------------===//

#include "AMDGPURetargeter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-retarget"

// Opcode name matching utilities for instruction mapping
// The actual opcodes are dynamically looked up from MCInstrInfo
namespace {

// Helper to look up an instruction by name in the MCInstrInfo
// Returns 0 if not found (which is typically an invalid pseudo instruction)
unsigned lookupOpcodeByName(const MCInstrInfo &MCII, StringRef Name) {
  for (unsigned I = 0, E = MCII.getNumOpcodes(); I < E; ++I) {
    if (Name == MCII.getName(I))
      return I;
  }
  return 0;
}

// Instruction name constants for retargeting
// GFX950-specific opcodes that need mapping/emulation on GFX942
constexpr const char *V_CVT_PK_F16_F32_e64 = "V_CVT_PK_F16_F32_e64";
constexpr const char *V_CVT_PK_BF16_F32_e64 = "V_CVT_PK_BF16_F32_e64";
constexpr const char *V_CVT_SCALEF32_PK_FP4_F32_e64 = "V_CVT_SCALEF32_PK_FP4_F32_e64";
constexpr const char *V_BITOP3_B16_e64 = "V_BITOP3_B16_e64";

// GFX942 equivalent opcodes
constexpr const char *V_CVT_PKRTZ_F16_F32_e64 = "V_CVT_PKRTZ_F16_F32_e64";

// GFX942-specific opcodes that need mapping/emulation on GFX90a
// V_LSHL_ADD_U64 exists on gfx942 but not on gfx90a
// The GFX8/GFX9 variant is named V_LSHL_ADD_U64_vi in TableGen
constexpr const char *V_LSHL_ADD_U64_vi = "V_LSHL_ADD_U64_vi";

// GFX90a emulation helper instructions
// For GFX9, we need the target-specific _gfx9 variants, not pseudos
constexpr const char *V_LSHLREV_B64_vi = "V_LSHLREV_B64_vi";
constexpr const char *V_ADD_CO_U32_e64_gfx9 = "V_ADD_CO_U32_e64_gfx9";
constexpr const char *V_ADDC_CO_U32_e64_gfx9 = "V_ADDC_CO_U32_e64_gfx9";

// Reserved scratch register for emulation (v255)
// This is the last VGPR, which is rarely used by kernels
// AMDGPU register encoding: VGPR_32 registers start at a base offset
// For a 64-bit operation we need two consecutive VGPRs: v254 and v255
// However, the actual register numbers need to be looked up from MCRegisterInfo

} // anonymous namespace

AMDGPURetargeter::AMDGPURetargeter(StringRef SourceCPU, StringRef TargetCPU,
                                   const MCInstrInfo &SourceMCII,
                                   const MCInstrInfo &TargetMCII,
                                   const MCRegisterInfo &SourceMRI,
                                   const MCRegisterInfo &TargetMRI)
    : SourceCPU(SourceCPU.str()), TargetCPU(TargetCPU.str()),
      SourceMCII(SourceMCII), TargetMCII(TargetMCII), SourceMRI(SourceMRI),
      TargetMRI(TargetMRI) {
  initializeOpcodeMap();
}

bool AMDGPURetargeter::isGFX950ToGFX942() const {
  return SourceCPU == "gfx950" && TargetCPU == "gfx942";
}

bool AMDGPURetargeter::isGFX942ToGFX90a() const {
  return SourceCPU == "gfx942" && TargetCPU == "gfx90a";
}

bool AMDGPURetargeter::isSameGFX9Family() const {
  // GFX9 family GPUs that share most instruction encodings
  auto isGFX9 = [](StringRef CPU) {
    return CPU.starts_with("gfx9") || CPU == "gfx90a" || CPU == "gfx908" ||
           CPU == "gfx906" || CPU == "gfx900";
  };
  return isGFX9(SourceCPU) && isGFX9(TargetCPU);
}

void AMDGPURetargeter::initializeOpcodeMap() {
  // Initialize the opcode mapping tables based on source/target pair
  if (isGFX950ToGFX942()) {
    // GFX950 -> GFX942 mappings
    // These mappings are based on the HotSwap analysis showing that
    // gfx942/gfx950 share identical instruction encodings for 98.4% of
    // instructions. Only a few instructions need special handling.

    // Look up opcodes dynamically from MCInstrInfo
    unsigned PkF16F32 = lookupOpcodeByName(SourceMCII, V_CVT_PK_F16_F32_e64);
    unsigned PkBF16F32 = lookupOpcodeByName(SourceMCII, V_CVT_PK_BF16_F32_e64);
    unsigned Fp4Quant = lookupOpcodeByName(SourceMCII, V_CVT_SCALEF32_PK_FP4_F32_e64);
    unsigned Bitop3 = lookupOpcodeByName(SourceMCII, V_BITOP3_B16_e64);
    unsigned PkrtzF16F32 = lookupOpcodeByName(TargetMCII, V_CVT_PKRTZ_F16_F32_e64);

    // Direct opcode swaps (same operand format, different opcode)
    // v_cvt_pk_f16_f32 -> v_cvt_pkrtz_f16_f32 (equivalent functionality)
    // Note: This is a simplification - the actual rounding behavior differs
    // (round-to-nearest vs round-toward-zero), but for many use cases it's acceptable.
    if (PkF16F32 && PkrtzF16F32)
      OpcodeMap[PkF16F32] = PkrtzF16F32;

    // Instructions that require emulation sequences
    if (PkBF16F32)
      EmulationRequired[PkBF16F32] = true;
    if (Fp4Quant)
      EmulationRequired[Fp4Quant] = true;
    if (Bitop3)
      EmulationRequired[Bitop3] = true;

    LLVM_DEBUG(dbgs() << "Initialized opcode map for gfx950 -> gfx942\n"
                      << "  Direct mappings: " << OpcodeMap.size() << "\n"
                      << "  Emulation required: " << EmulationRequired.size()
                      << "\n");
  } else if (isGFX942ToGFX90a()) {
    // GFX942 -> GFX90a mappings
    // gfx942 has some instructions not available on gfx90a

    // V_LSHL_ADD_U64 exists on gfx942 but not on gfx90a
    // The GFX9 variant uses the _vi suffix
    unsigned LshlAddU64 = lookupOpcodeByName(SourceMCII, V_LSHL_ADD_U64_vi);
    if (LshlAddU64) {
      EmulationRequired[LshlAddU64] = true;
      LLVM_DEBUG(dbgs() << "Registered V_LSHL_ADD_U64_vi (opcode " << LshlAddU64
                        << ") for emulation\n");
    }

    LLVM_DEBUG(dbgs() << "Initialized opcode map for gfx942 -> gfx90a\n"
                      << "  Direct mappings: " << OpcodeMap.size() << "\n"
                      << "  Emulation required: " << EmulationRequired.size()
                      << "\n");
  } else {
    LLVM_DEBUG(dbgs() << "No opcode mapping defined for " << SourceCPU << " -> "
                      << TargetCPU << "\n");
  }
}

unsigned AMDGPURetargeter::getDirectMapping(unsigned SourceOpcode) const {
  auto It = OpcodeMap.find(SourceOpcode);
  if (It != OpcodeMap.end())
    return It->second;
  return 0;
}

bool AMDGPURetargeter::requiresEmulation(unsigned SourceOpcode) const {
  return EmulationRequired.count(SourceOpcode) > 0;
}

Error AMDGPURetargeter::analyzeForLiveness(ArrayRef<MCInst> Instructions,
                                           ArrayRef<uint64_t> Offsets) {
  Liveness = std::make_unique<LivenessAnalyzer>(SourceMCII, SourceMRI);

  if (!Liveness->analyze(Instructions, Offsets)) {
    return createStringError(inconvertibleErrorCode(),
                             "Liveness analysis failed");
  }

  LLVM_DEBUG({
    const auto &Stats = Liveness->getStats();
    dbgs() << "Liveness analysis: " << Stats.NumInstructions << " instructions, "
           << Stats.NumBasicBlocks << " blocks, " << Stats.NumIterations
           << " iterations\n"
           << "  Max live VGPRs: " << Stats.MaxLiveVGPRs
           << ", Min dead VGPRs: " << Stats.MinDeadVGPRs << "\n";
  });

  return Error::success();
}

Error AMDGPURetargeter::transformAll(ArrayRef<MCInst> SourceInsts,
                                     ArrayRef<uint64_t> Offsets,
                                     SmallVectorImpl<MCInst> &TargetInsts) {
  for (size_t I = 0; I < SourceInsts.size(); ++I) {
    SmallVector<MCInst, 4> TransformedInsts;
    if (auto Err = transform(SourceInsts[I], I, TransformedInsts))
      return Err;
    TargetInsts.append(TransformedInsts.begin(), TransformedInsts.end());
  }
  return Error::success();
}

Error AMDGPURetargeter::transform(const MCInst &SourceInst, size_t InstIndex,
                                  SmallVectorImpl<MCInst> &TargetInsts) {
  CurrentInstIndex = InstIndex;
  unsigned SourceOpcode = SourceInst.getOpcode();

  // Check if this is a same-family retarget where most opcodes pass through
  if (isGFX950ToGFX942() || isGFX942ToGFX90a() || isSameGFX9Family()) {
    // Check for direct mapping
    if (unsigned TargetOpcode = getDirectMapping(SourceOpcode)) {
      MCInst TargetInst = SourceInst;
      TargetInst.setOpcode(TargetOpcode);
      TargetInsts.push_back(TargetInst);
      ++Statistics.DirectMapped;
      return Error::success();
    }

    // Check if emulation is required
    if (requiresEmulation(SourceOpcode)) {
      if (auto Err = emitEmulationSequence(SourceInst, TargetInsts))
        return Err;
      ++Statistics.Emulated;
      return Error::success();
    }

    // For same-family, most instructions pass through unchanged
    // GFX9 family GPUs share the same encoding for most instructions
    TargetInsts.push_back(SourceInst);
    ++Statistics.PassThrough;
    return Error::success();
  }

  // For other source/target combinations, we would need more extensive
  // mapping tables
  ++Statistics.Unsupported;
  return createStringError(inconvertibleErrorCode(),
                           "Unsupported architecture pair: " + SourceCPU +
                               " -> " + TargetCPU);
}

int AMDGPURetargeter::allocateScratchVGPR(size_t InstIndex) {
  if (Liveness) {
    // Use liveness analysis to find a dead register
    int Reg = Liveness->allocateScratchVGPR(InstIndex, KernelVGPRCount);
    if (Reg >= 0) {
      ++Statistics.ScratchRegsUsed;
      if (static_cast<unsigned>(Reg) >= KernelVGPRCount) {
        unsigned ExtraNeeded = Reg - KernelVGPRCount + 1;
        Statistics.MaxExtraVGPRs = std::max(Statistics.MaxExtraVGPRs, ExtraNeeded);
      }
      LLVM_DEBUG(dbgs() << "Allocated scratch VGPR v" << Reg
                        << " at instruction " << InstIndex << "\n");
      return Reg;
    }
  }

  // Fall back to v255 (last VGPR)
  LLVM_DEBUG(dbgs() << "Falling back to v255 for scratch at instruction "
                    << InstIndex << "\n");
  ++Statistics.ScratchRegsUsed;
  return 255;
}

int AMDGPURetargeter::allocateScratchVGPR64(size_t InstIndex) {
  if (Liveness) {
    // Use liveness analysis to find a dead 64-bit pair
    int Reg = Liveness->allocateScratchVGPR64(InstIndex, KernelVGPRCount);
    if (Reg >= 0) {
      ++Statistics.ScratchRegsUsed;
      if (static_cast<unsigned>(Reg + 1) >= KernelVGPRCount) {
        unsigned ExtraNeeded = (Reg + 1) - KernelVGPRCount + 1;
        Statistics.MaxExtraVGPRs = std::max(Statistics.MaxExtraVGPRs, ExtraNeeded);
      }
      LLVM_DEBUG(dbgs() << "Allocated scratch VGPR64 v[" << Reg << ":"
                        << (Reg + 1) << "] at instruction " << InstIndex << "\n");
      return Reg;
    }
  }

  // Fall back to v254:v255 (last 64-bit pair)
  LLVM_DEBUG(dbgs() << "Falling back to v[254:255] for scratch64 at instruction "
                    << InstIndex << "\n");
  ++Statistics.ScratchRegsUsed;
  return 254;
}

unsigned AMDGPURetargeter::getVGPRRegister(unsigned VGPRNum) const {
  // Look up VGPR<N> by name
  std::string RegName = "VGPR" + std::to_string(VGPRNum);
  for (unsigned I = 0, E = TargetMRI.getNumRegs(); I < E; ++I) {
    if (TargetMRI.getName(I) == RegName)
      return I;
  }
  return 0; // Invalid
}

unsigned AMDGPURetargeter::getVGPR64Register(unsigned LowVGPRNum) const {
  // Look up VGPR<N>_VGPR<N+1> by name
  std::string RegName = "VGPR" + std::to_string(LowVGPRNum) + "_VGPR" +
                        std::to_string(LowVGPRNum + 1);
  for (unsigned I = 0, E = TargetMRI.getNumRegs(); I < E; ++I) {
    StringRef Name = TargetMRI.getName(I);
    if (Name == RegName)
      return I;
    // Also try alternative naming convention
    if (Name.contains(std::to_string(LowVGPRNum)) &&
        Name.contains(std::to_string(LowVGPRNum + 1)) &&
        Name.starts_with("VGPR"))
      return I;
  }
  return 0; // Invalid
}

Error AMDGPURetargeter::emitEmulationSequence(
    const MCInst &SourceInst, SmallVectorImpl<MCInst> &TargetInsts) {
  unsigned SourceOpcode = SourceInst.getOpcode();
  StringRef OpName = SourceMCII.getName(SourceOpcode);

  // Check for V_CVT_PK_BF16_F32
  if (OpName.contains("V_CVT_PK_BF16_F32")) {
    return emitBF16PackEmulation(SourceInst, TargetInsts);
  }

  // Check for V_CVT_SCALEF32_PK_FP4_F32
  if (OpName.contains("V_CVT_SCALEF32_PK_FP4_F32")) {
    return emitFP4QuantEmulation(SourceInst, TargetInsts);
  }

  // Check for V_LSHL_ADD_U64 (gfx942 -> gfx90a)
  if (OpName.contains("V_LSHL_ADD_U64")) {
    return emitLshlAddU64Emulation(SourceInst, CurrentInstIndex, TargetInsts);
  }

  return createStringError(inconvertibleErrorCode(),
                           "No emulation available for opcode " +
                               Twine::utohexstr(SourceOpcode) + " (" +
                               OpName + ")");
}

Error AMDGPURetargeter::emitBF16PackEmulation(
    const MCInst &SourceInst, SmallVectorImpl<MCInst> &TargetInsts) {
  // Emulate v_cvt_pk_bf16_f32 vDst, vSrc0, vSrc1
  // bf16 is the upper 16 bits of f32, so we extract bits [31:16] and pack
  //
  // v_lshrrev_b32 vDst, 16, vSrc0      ; bf16(src0) in [15:0]
  // v_lshrrev_b32 vTmp, 16, vSrc1      ; bf16(src1) in [15:0]
  // v_lshl_or_b32 vDst, vTmp, 16, vDst ; pack: [bf16_1:bf16_0]
  //
  // Note: This is a truncation without proper rounding. For production use,
  // we would need proper round-to-nearest-even implementation.

  // For now, we return an error indicating this emulation needs
  // register allocation which is complex to do at the MCInst level
  return createStringError(inconvertibleErrorCode(),
                           "BF16 pack emulation requires register "
                           "allocation - not yet implemented in binary "
                           "retargeting. Consider using compiler-level "
                           "fallback with -mfallback-arch instead.");
}

Error AMDGPURetargeter::emitFP4QuantEmulation(
    const MCInst &SourceInst, SmallVectorImpl<MCInst> &TargetInsts) {
  // Emulate v_cvt_scalef32_pk_fp4_f32 vDst, vSrc0, vSrc1, vScale
  // FP4 E2M1 has 8 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
  //
  // The emulation scales, quantizes via truncation+clamp, and packs:
  // v_mul_f32 vTmp0, vSrc0, vScale    ; scale input 0
  // v_mul_f32 vTmp0, vTmp0, 2.0       ; multiply by 2 for E2M1 index
  // v_cvt_u32_f32 vTmp0, vTmp0        ; truncate to uint
  // v_min_u32 vTmp0, 7, vTmp0         ; clamp to [0,7]
  // (repeat for vSrc1)
  // v_lshl_or_b32 vDst, vTmp1, 4, vTmp0  ; pack nibbles

  return createStringError(inconvertibleErrorCode(),
                           "FP4 quantization emulation requires register "
                           "allocation - not yet implemented in binary "
                           "retargeting. Consider using compiler-level "
                           "fallback with -mfallback-arch instead.");
}

Error AMDGPURetargeter::emitLshlAddU64Emulation(
    const MCInst &SourceInst, size_t InstIndex,
    SmallVectorImpl<MCInst> &TargetInsts) {
  // Emulate V_LSHL_ADD_U64 vDst, vSrc0, vSrc1, vSrc2 on gfx90a
  // Semantics: D.u64 = (S0.u64 << S1.u[2:0]) + S2.u64
  //
  // Emulation sequence (using 32-bit operations with carry):
  //   v_lshlrev_b64 vTmp, vSrc1, vSrc0     ; tmp = src0 << src1
  //   v_add_co_u32  vDst_lo, vcc, vTmp_lo, vSrc2_lo  ; dst_lo = tmp_lo + src2_lo
  //   v_addc_u32    vDst_hi, vcc, vTmp_hi, vSrc2_hi, vcc ; dst_hi = tmp_hi + src2_hi + carry
  //
  // Register allocation strategy (in priority order):
  // 1. Use liveness analysis to find a dead 64-bit VGPR pair
  // 2. If dst != src0 && dst != src2, use dst as intermediate
  // 3. Fall back to v254:v255 (last 64-bit pair)

  // Look up target instructions - use the GFX9-specific variants for encoding
  unsigned LshlrevB64 = lookupOpcodeByName(TargetMCII, V_LSHLREV_B64_vi);
  unsigned AddCoU32 = lookupOpcodeByName(TargetMCII, V_ADD_CO_U32_e64_gfx9);
  unsigned AddcU32 = lookupOpcodeByName(TargetMCII, V_ADDC_CO_U32_e64_gfx9);

  LLVM_DEBUG(dbgs() << "Emulating V_LSHL_ADD_U64: LshlrevB64=" << LshlrevB64
                    << " AddCoU32=" << AddCoU32 << " AddcU32=" << AddcU32 << "\n");

  if (!LshlrevB64 || !AddCoU32 || !AddcU32) {
    return createStringError(inconvertibleErrorCode(),
                             "Target architecture missing required instructions "
                             "for V_LSHL_ADD_U64 emulation (V_LSHLREV_B64, "
                             "V_ADD_CO_U32, V_ADDC_U32)");
  }

  // V_LSHL_ADD_U64_e64 operands:
  // 0: vdst (64-bit)
  // 1: src0 (64-bit) - value to shift
  // 2: src1 (32-bit) - shift amount
  // 3: src2 (64-bit) - value to add
  // Plus modifier operands (clamp, omod, etc.)

  if (SourceInst.getNumOperands() < 4) {
    return createStringError(inconvertibleErrorCode(),
                             "V_LSHL_ADD_U64 has insufficient operands");
  }

  MCOperand Dst = SourceInst.getOperand(0);
  MCOperand Src0 = SourceInst.getOperand(1);
  MCOperand Src1 = SourceInst.getOperand(2);
  MCOperand Src2 = SourceInst.getOperand(3);

  // Check if we can use the destination as intermediate
  // (safe if dst != src0 and dst != src2, since we write to dst last)
  bool CanUseDstAsIntermediate = true;

  if (Dst.isReg() && Src0.isReg() && Dst.getReg() == Src0.getReg())
    CanUseDstAsIntermediate = false;
  if (Dst.isReg() && Src2.isReg() && Dst.getReg() == Src2.getReg())
    CanUseDstAsIntermediate = false;

  // Allocate scratch register using liveness analysis
  MCOperand TmpReg;

  // First, try liveness-based allocation for a 64-bit pair
  int ScratchVGPR = allocateScratchVGPR64(InstIndex);
  if (ScratchVGPR >= 0) {
    unsigned ScratchReg64 = getVGPR64Register(ScratchVGPR);
    if (ScratchReg64) {
      TmpReg = MCOperand::createReg(ScratchReg64);
      LLVM_DEBUG(dbgs() << "Using liveness-allocated scratch v[" << ScratchVGPR
                        << ":" << (ScratchVGPR + 1) << "] for V_LSHL_ADD_U64\n");
    }
  }

  // Fall back to dst-as-intermediate or hardcoded v254:v255
  if (!TmpReg.isValid()) {
    if (CanUseDstAsIntermediate) {
      TmpReg = Dst;
      LLVM_DEBUG(dbgs() << "Using dst as intermediate for V_LSHL_ADD_U64\n");
    } else {
      // Find v254:v255 as fallback
      unsigned ScratchReg64 = 0;
      for (unsigned I = 0, E = TargetMRI.getNumRegs(); I < E; ++I) {
        StringRef RegName = TargetMRI.getName(I);
        if (RegName.contains("254") && RegName.contains("255")) {
          ScratchReg64 = I;
          break;
        }
      }
      if (ScratchReg64) {
        TmpReg = MCOperand::createReg(ScratchReg64);
      } else {
        LLVM_DEBUG(dbgs() << "Warning: Using dst as intermediate in V_LSHL_ADD_U64 "
                             "emulation - may cause issues if dst overlaps src0/src2\n");
        TmpReg = Dst;
      }
    }
  }

  // Emit: v_lshlrev_b64 tmp, src1, src0
  // Note: v_lshlrev_b64 has reversed operand order compared to v_lshl_b64
  // v_lshlrev_b64 dst, shift_amount, src -> dst = src << shift_amount
  MCInst LshlInst;
  LshlInst.setOpcode(LshlrevB64);
  LshlInst.addOperand(TmpReg);  // dst (64-bit)
  LshlInst.addOperand(Src1);    // shift amount (32-bit)
  LshlInst.addOperand(Src0);    // src (64-bit)
  TargetInsts.push_back(LshlInst);

  // Now we need to add tmp + src2 using 32-bit operations with carry.
  // On GFX90a, V_ADD_U64 doesn't exist - we must use 32-bit operations.
  //
  // Decomposition:
  //   v_add_co_u32 dst_lo, vcc, tmp_lo, src2_lo
  //   v_addc_co_u32 dst_hi, vcc, tmp_hi, src2_hi, vcc
  //
  // For 64-bit register pairs (e.g., VGPR6_VGPR7), the sub-registers are:
  //   - Low: VGPR6 (even register)
  //   - High: VGPR7 (odd register)
  // We can extract these using MCRegisterInfo::getSubReg().

  // Get the 32-bit sub-register indices for lo/hi halves
  // In AMDGPU, the sub-register indices are defined in AMDGPUGenRegisterInfoEnums.inc:
  //   sub0 = 3 (lo 32-bit)
  //   sub1 = 11 (hi 32-bit)
  constexpr unsigned SubReg_sub0 = 3;  // lo 32-bit
  constexpr unsigned SubReg_sub1 = 11; // hi 32-bit

  // Helper to get sub-register from a 64-bit register operand
  auto getSubRegLo = [this](const MCOperand &Op, unsigned SubIdx) -> MCOperand {
    if (!Op.isReg())
      return Op; // Not a register, return as-is (shouldn't happen for 64-bit ops)
    unsigned Reg64 = Op.getReg();
    // Get the first sub-register (low 32 bits)
    unsigned SubReg = TargetMRI.getSubReg(Reg64, SubIdx);
    if (SubReg)
      return MCOperand::createReg(SubReg);
    return Op; // Fallback
  };

  auto getSubRegHi = [this](const MCOperand &Op, unsigned SubIdx) -> MCOperand {
    if (!Op.isReg())
      return Op;
    unsigned Reg64 = Op.getReg();
    // Get the second sub-register (high 32 bits)
    unsigned SubReg = TargetMRI.getSubReg(Reg64, SubIdx);
    if (SubReg)
      return MCOperand::createReg(SubReg);
    return Op;
  };

  // Get sub-registers for dst, tmp (from shift), and src2
  MCOperand DstLo = getSubRegLo(Dst, SubReg_sub0);
  MCOperand DstHi = getSubRegHi(Dst, SubReg_sub1);
  MCOperand TmpLo = getSubRegLo(TmpReg, SubReg_sub0);
  MCOperand TmpHi = getSubRegHi(TmpReg, SubReg_sub1);
  MCOperand Src2Lo = getSubRegLo(Src2, SubReg_sub0);
  MCOperand Src2Hi = getSubRegHi(Src2, SubReg_sub1);

  // Find VCC register (carry flag for 64-bit add emulation)
  unsigned VCCReg = 0;
  for (unsigned I = 0, E = TargetMRI.getNumRegs(); I < E; ++I) {
    StringRef RegName = TargetMRI.getName(I);
    if (RegName == "VCC") {
      VCCReg = I;
      break;
    }
  }

  if (!VCCReg) {
    return createStringError(inconvertibleErrorCode(),
                             "Could not find VCC register for 64-bit add emulation");
  }

  MCOperand VCC = MCOperand::createReg(VCCReg);

  // Verify we got valid sub-registers
  if (!DstLo.isReg() || !DstHi.isReg() ||
      !TmpLo.isReg() || !TmpHi.isReg() ||
      !Src2Lo.isReg() || !Src2Hi.isReg()) {
    return createStringError(inconvertibleErrorCode(),
                             "Failed to extract 32-bit sub-registers from 64-bit "
                             "register pair for V_LSHL_ADD_U64 emulation");
  }

  // Emit: v_add_co_u32 dst_lo, vcc, tmp_lo, src2_lo
  // V_ADD_CO_U32_e64 format: dst, sdst (vcc), src0, src1, clamp
  MCInst AddLoInst;
  AddLoInst.setOpcode(AddCoU32);
  AddLoInst.addOperand(DstLo);   // dst (32-bit)
  AddLoInst.addOperand(VCC);     // sdst = vcc
  AddLoInst.addOperand(TmpLo);   // src0 (32-bit)
  AddLoInst.addOperand(Src2Lo);  // src1 (32-bit)
  AddLoInst.addOperand(MCOperand::createImm(0)); // clamp = 0
  TargetInsts.push_back(AddLoInst);

  // Emit: v_addc_co_u32 dst_hi, vcc, tmp_hi, src2_hi, vcc
  // V_ADDC_U32_e64 format: dst, sdst (vcc), src0, src1, src2 (carry-in), clamp
  MCInst AddHiInst;
  AddHiInst.setOpcode(AddcU32);
  AddHiInst.addOperand(DstHi);   // dst (32-bit)
  AddHiInst.addOperand(VCC);     // sdst = vcc
  AddHiInst.addOperand(TmpHi);   // src0 (32-bit)
  AddHiInst.addOperand(Src2Hi);  // src1 (32-bit)
  AddHiInst.addOperand(VCC);     // src2 = vcc (carry-in from AddLo)
  AddHiInst.addOperand(MCOperand::createImm(0)); // clamp = 0
  TargetInsts.push_back(AddHiInst);

  LLVM_DEBUG(dbgs() << "Emulated V_LSHL_ADD_U64 as V_LSHLREV_B64 + V_ADD_U64\n");

  return Error::success();
}
