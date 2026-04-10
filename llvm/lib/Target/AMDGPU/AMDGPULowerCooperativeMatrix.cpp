//===-- AMDGPULowerCooperativeMatrix.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass lowers cooperative matrix intrinsics to AMDGPU target-specific
/// intrinsics (MFMA/WMMA). Cooperative matrix TargetExtTypes have no MVT/LLT
/// representation and must be rewritten to concrete LLVM vector types before
/// entering SelectionDAG or GlobalISel.
///
/// The pass rewrites all uses of cooperative matrix values, replacing
/// TargetExtType values with concrete vector values throughout.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-cooperative-matrix"

namespace {

/// Cooperative matrix "use" values from the SPIR-V / intrinsic convention.
enum CoopMatUse : unsigned {
  MatrixA = 0,
  MatrixB = 1,
  Accumulator = 2,
};

/// Describes how to lower a cooperative matrix muladd to a target intrinsic.
struct MulAddEntry {
  Intrinsic::ID IntrID;
  Type *ABVecTy;  // Vector type for A and B operands
  Type *CDVecTy;  // Vector type for C/D (accumulator)
  bool IsMFMA;    // true = MFMA (has cbsz/abid/blgp), false = WMMA
  bool IsIU;      // true = integer unsigned (IU intrinsic pattern)
};

/// Extract cooperative matrix parameters from a TargetExtType.
/// Format: target("spirv.CooperativeMatrixKHR", elem, scope, rows, cols, use)
static bool getCoopMatParams(Type *Ty, Type *&ElemTy, unsigned &Scope,
                             unsigned &Rows, unsigned &Cols, unsigned &Use) {
  auto *TET = dyn_cast<TargetExtType>(Ty);
  if (!TET || TET->getName() != "spirv.CooperativeMatrixKHR")
    return false;
  if (TET->getNumTypeParameters() < 1 || TET->getNumIntParameters() < 4)
    return false;
  ElemTy = TET->getTypeParameter(0);
  Scope = TET->getIntParameter(0);
  Rows = TET->getIntParameter(1);
  Cols = TET->getIntParameter(2);
  Use = TET->getIntParameter(3);
  return true;
}

/// Get the concrete vector type that a cooperative matrix maps to on a given
/// subtarget, based on element type, dimensions, and use.
static FixedVectorType *getConcreteVectorType(const GCNSubtarget &ST,
                                              Type *ElemTy, unsigned Rows,
                                              unsigned Cols, unsigned Use) {
  // WMMA v2/v3 (wave32, gfx12+): 8 elements per lane for all matrix operands.
  if (ST.hasWMMA128bInsts() && ST.isWave32()) {
    if (ElemTy->isHalfTy() || ElemTy->isBFloatTy())
      return FixedVectorType::get(ElemTy, 8);
    if (ElemTy->isFloatTy())
      return FixedVectorType::get(ElemTy, 8);
    if (ElemTy->isIntegerTy(8))
      return FixedVectorType::get(ElemTy, 8);
    if (ElemTy->isIntegerTy(32))
      return FixedVectorType::get(ElemTy, 8);
  }

  // WMMA v1 (wave32, gfx11): 256-bit A/B (duplicated), 256-bit C/D.
  // Same intrinsic types as v2 at the IR level: 8 elements per lane.
  if (ST.hasWMMA256bInsts() && ST.isWave32()) {
    if (ElemTy->isHalfTy() || ElemTy->isBFloatTy())
      return FixedVectorType::get(ElemTy, 8);
    if (ElemTy->isFloatTy())
      return FixedVectorType::get(ElemTy, 8);
    if (ElemTy->isIntegerTy(8))
      return FixedVectorType::get(ElemTy, 8);
    if (ElemTy->isIntegerTy(32))
      return FixedVectorType::get(ElemTy, 8);
  }

  // MFMA (wave64, gfx908+)
  if (ST.hasMAIInsts() && ST.isWave64()) {
    if (Use == Accumulator) {
      // Accumulator dimensions determine the vector length.
      if (ElemTy->isFloatTy()) {
        if (Rows == 16 && Cols == 16)
          return FixedVectorType::get(ElemTy, 4);
        if (Rows == 32 && Cols == 32)
          return FixedVectorType::get(ElemTy, 16);
      }
      if (ElemTy->isIntegerTy(32)) {
        if (Rows == 16 && Cols == 16)
          return FixedVectorType::get(ElemTy, 4);
        if (Rows == 32 && Cols == 32)
          return FixedVectorType::get(ElemTy, 16);
      }
      // f64 accumulator: 16x16 -> <4 x f64>, 4x4 -> scalar (1-elem vec).
      if (ElemTy->isDoubleTy()) {
        if (Rows == 16 && Cols == 16)
          return FixedVectorType::get(ElemTy, 4);
        if (Rows == 4 && Cols == 4)
          return FixedVectorType::get(ElemTy, 1);
      }
    } else {
      // A/B operands for MFMA. Per-lane element count = (Rows * Cols) / 64.
      unsigned TotalElems = Rows * Cols;
      unsigned VecLen = TotalElems / 64;
      if (VecLen == 0) VecLen = 1; // f64 4x4x4: scalar per lane
      if (ElemTy->isHalfTy() || ElemTy->isBFloatTy())
        return FixedVectorType::get(ElemTy, VecLen);
      if (ElemTy->isIntegerTy(8))
        return FixedVectorType::get(ElemTy, VecLen);
      if (ElemTy->isDoubleTy())
        return FixedVectorType::get(ElemTy, VecLen);
    }
  }

  report_fatal_error("Unsupported cooperative matrix configuration for AMDGPU");
}

/// Look up the muladd lowering for a given configuration.
static std::optional<MulAddEntry>
lookupMulAdd(const GCNSubtarget &ST, LLVMContext &Ctx,
             Type *ElemTyA, Type *ElemTyB, Type *ElemTyC,
             unsigned M, unsigned N, unsigned K) {
  MulAddEntry Entry;

  // --- WMMA (gfx12, wave32) ---
  if (ST.hasWMMA128bInsts() && ST.isWave32()) {
    // f16 * f16 + f32 -> f32, 16x16x16
    if (ElemTyA->isHalfTy() && ElemTyB->isHalfTy() && ElemTyC->isFloatTy() &&
        M == 16 && N == 16 && K == 16) {
      Entry.IntrID = Intrinsic::amdgcn_wmma_f32_16x16x16_f16;
      Entry.ABVecTy = FixedVectorType::get(Type::getHalfTy(Ctx), 8);
      Entry.CDVecTy = FixedVectorType::get(Type::getFloatTy(Ctx), 8);
      Entry.IsMFMA = false;
      Entry.IsIU = false;
      return Entry;
    }
    // i8 * i8 + i32 -> i32, 16x16x16
    if (ElemTyA->isIntegerTy(8) && ElemTyB->isIntegerTy(8) &&
        ElemTyC->isIntegerTy(32) && M == 16 && N == 16 && K == 16) {
      // The intrinsic takes <2 x i32> for A/B (packed i8 values).
      Entry.IntrID = Intrinsic::amdgcn_wmma_i32_16x16x16_iu8;
      Entry.ABVecTy = FixedVectorType::get(Type::getInt32Ty(Ctx), 2);
      Entry.CDVecTy = FixedVectorType::get(Type::getInt32Ty(Ctx), 8);
      Entry.IsMFMA = false;
      Entry.IsIU = true;
      return Entry;
    }
  }

  // --- WMMA v1 (gfx11, wave32) ---
  // Same intrinsic names as v2 but with different HW register sizes (256-bit
  // A/B with duplicated data). At the IR intrinsic level the types are identical
  // to v2: <8 x f16> for A/B, <8 x f32> for C/D.
  if (ST.hasWMMA256bInsts() && !ST.hasWMMA128bInsts() && ST.isWave32()) {
    // f16 * f16 + f32 -> f32, 16x16x16
    if (ElemTyA->isHalfTy() && ElemTyB->isHalfTy() && ElemTyC->isFloatTy() &&
        M == 16 && N == 16 && K == 16) {
      Entry.IntrID = Intrinsic::amdgcn_wmma_f32_16x16x16_f16;
      Entry.ABVecTy = FixedVectorType::get(Type::getHalfTy(Ctx), 8);
      Entry.CDVecTy = FixedVectorType::get(Type::getFloatTy(Ctx), 8);
      Entry.IsMFMA = false;
      Entry.IsIU = false;
      return Entry;
    }
    // bf16 * bf16 + f32 -> f32, 16x16x16
    if (ElemTyA->isBFloatTy() && ElemTyB->isBFloatTy() &&
        ElemTyC->isFloatTy() && M == 16 && N == 16 && K == 16) {
      Entry.IntrID = Intrinsic::amdgcn_wmma_f32_16x16x16_bf16;
      // bf16 WMMA intrinsic takes <8 x i16> (bitcast of bf16)
      Entry.ABVecTy = FixedVectorType::get(Type::getInt16Ty(Ctx), 8);
      Entry.CDVecTy = FixedVectorType::get(Type::getFloatTy(Ctx), 8);
      Entry.IsMFMA = false;
      Entry.IsIU = false;
      return Entry;
    }
    // i8 * i8 + i32 -> i32, 16x16x16
    if (ElemTyA->isIntegerTy(8) && ElemTyB->isIntegerTy(8) &&
        ElemTyC->isIntegerTy(32) && M == 16 && N == 16 && K == 16) {
      Entry.IntrID = Intrinsic::amdgcn_wmma_i32_16x16x16_iu8;
      Entry.ABVecTy = FixedVectorType::get(Type::getInt32Ty(Ctx), 2);
      Entry.CDVecTy = FixedVectorType::get(Type::getInt32Ty(Ctx), 8);
      Entry.IsMFMA = false;
      Entry.IsIU = true;
      return Entry;
    }
  }

  // --- MFMA (gfx908, wave64) ---
  if (ST.hasMAIInsts() && ST.isWave64()) {
    // f16 * f16 + f32 -> f32, 16x16x16
    if (ElemTyA->isHalfTy() && ElemTyB->isHalfTy() && ElemTyC->isFloatTy() &&
        M == 16 && N == 16 && K == 16) {
      Entry.IntrID = Intrinsic::amdgcn_mfma_f32_16x16x16f16;
      Entry.ABVecTy = FixedVectorType::get(Type::getHalfTy(Ctx), 4);
      Entry.CDVecTy = FixedVectorType::get(Type::getFloatTy(Ctx), 4);
      Entry.IsMFMA = true;
      Entry.IsIU = false;
      return Entry;
    }
    // f16 * f16 + f32 -> f32, 32x32x8
    if (ElemTyA->isHalfTy() && ElemTyB->isHalfTy() && ElemTyC->isFloatTy() &&
        M == 32 && N == 32 && K == 8) {
      Entry.IntrID = Intrinsic::amdgcn_mfma_f32_32x32x8f16;
      Entry.ABVecTy = FixedVectorType::get(Type::getHalfTy(Ctx), 4);
      Entry.CDVecTy = FixedVectorType::get(Type::getFloatTy(Ctx), 16);
      Entry.IsMFMA = true;
      Entry.IsIU = false;
      return Entry;
    }
    // bf16 * bf16 + f32 -> f32, 16x16x16 (gfx90a+, 1k variant)
    // BUG-6 fix: the _1k bf16 MFMA intrinsics only exist on gfx90a and later.
    // gfx908 has MAI instructions but no GFX90A-specific insts, so gate on
    // hasGFX90AInsts() here (not hasMAIInsts() which includes gfx908).
    if (ST.hasGFX90AInsts() && ElemTyA->isBFloatTy() && ElemTyB->isBFloatTy() &&
        ElemTyC->isFloatTy() && M == 16 && N == 16 && K == 16) {
      Entry.IntrID = Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k;
      Entry.ABVecTy = FixedVectorType::get(Type::getBFloatTy(Ctx), 4);
      Entry.CDVecTy = FixedVectorType::get(Type::getFloatTy(Ctx), 4);
      Entry.IsMFMA = true;
      Entry.IsIU = false;
      return Entry;
    }
    // bf16 * bf16 + f32 -> f32, 32x32x8 (gfx90a+, 1k variant)
    if (ST.hasGFX90AInsts() && ElemTyA->isBFloatTy() && ElemTyB->isBFloatTy() &&
        ElemTyC->isFloatTy() && M == 32 && N == 32 && K == 8) {
      Entry.IntrID = Intrinsic::amdgcn_mfma_f32_32x32x4bf16_1k;
      Entry.ABVecTy = FixedVectorType::get(Type::getBFloatTy(Ctx), 4);
      Entry.CDVecTy = FixedVectorType::get(Type::getFloatTy(Ctx), 16);
      Entry.IsMFMA = true;
      Entry.IsIU = false;
      return Entry;
    }
    // i8 * i8 + i32 -> i32, 16x16x16
    if (ElemTyA->isIntegerTy(8) && ElemTyB->isIntegerTy(8) &&
        ElemTyC->isIntegerTy(32) && M == 16 && N == 16 && K == 16) {
      Entry.IntrID = Intrinsic::amdgcn_mfma_i32_16x16x16i8;
      Entry.ABVecTy = FixedVectorType::get(Type::getInt8Ty(Ctx), 4);
      Entry.CDVecTy = FixedVectorType::get(Type::getInt32Ty(Ctx), 4);
      Entry.IsMFMA = true;
      Entry.IsIU = false;
      return Entry;
    }
    // i8 * i8 + i32 -> i32, 32x32x8
    if (ElemTyA->isIntegerTy(8) && ElemTyB->isIntegerTy(8) &&
        ElemTyC->isIntegerTy(32) && M == 32 && N == 32 && K == 8) {
      Entry.IntrID = Intrinsic::amdgcn_mfma_i32_32x32x8i8;
      Entry.ABVecTy = FixedVectorType::get(Type::getInt8Ty(Ctx), 4);
      Entry.CDVecTy = FixedVectorType::get(Type::getInt32Ty(Ctx), 16);
      Entry.IsMFMA = true;
      Entry.IsIU = false;
      return Entry;
    }

    // --- FP8/BF8 MFMA (gfx942 FNUZ / gfx950+ OCP) ---
    // The same intrinsic names (mfma_f32_16x16x32_fp8_fp8 etc.) are used
    // on both gfx942 and gfx950+; the hardware reinterprets bits as FNUZ
    // vs OCP. The cooperative matrix type's element type encodes which
    // format the user intended. We gate on hasFP8ConversionInsts() for
    // the HW capability and accept either format at this level — the
    // distinction is in the TargetExtType, not the intrinsic.
    if (ST.hasFP8ConversionInsts()) {
      auto *FloatTy = Type::getFloatTy(Ctx);
      // A/B operands are i64 in the MFMA intrinsic = 8 packed i8 values.
      // The concrete per-lane vector is <8 x i8>; CreateBitCast(<8 x i8>, i64)
      // is legal in LLVM IR (same total bit width).
      auto *I64Ty = Type::getInt64Ty(Ctx);

      // fp8 * fp8 + f32 -> f32, 16x16x32
      if (ElemTyA->isIntegerTy(8) && ElemTyB->isIntegerTy(8) &&
          ElemTyC->isFloatTy() && M == 16 && N == 16 && K == 32) {
        Entry.IntrID = Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8;
        Entry.ABVecTy = I64Ty;
        Entry.CDVecTy = FixedVectorType::get(FloatTy, 4);
        Entry.IsMFMA = true;
        Entry.IsIU = false;
        return Entry;
      }
      // fp8 * fp8 + f32 -> f32, 32x32x16
      if (ElemTyA->isIntegerTy(8) && ElemTyB->isIntegerTy(8) &&
          ElemTyC->isFloatTy() && M == 32 && N == 32 && K == 16) {
        Entry.IntrID = Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_fp8;
        Entry.ABVecTy = I64Ty;
        Entry.CDVecTy = FixedVectorType::get(FloatTy, 16);
        Entry.IsMFMA = true;
        Entry.IsIU = false;
        return Entry;
      }
    }

    // --- Larger f16 MFMA shapes (gfx942+) ---
    if (ST.hasGFX940Insts()) {
      // f16 * f16 + f32 -> f32, 16x16x32 (K=32)
      if (ElemTyA->isHalfTy() && ElemTyB->isHalfTy() &&
          ElemTyC->isFloatTy() && M == 16 && N == 16 && K == 32) {
        Entry.IntrID = Intrinsic::amdgcn_mfma_f32_16x16x32_f16;
        Entry.ABVecTy = FixedVectorType::get(Type::getHalfTy(Ctx), 8);
        Entry.CDVecTy = FixedVectorType::get(Type::getFloatTy(Ctx), 4);
        Entry.IsMFMA = true;
        Entry.IsIU = false;
        return Entry;
      }
      // f16 * f16 + f32 -> f32, 32x32x16 (K=16)
      if (ElemTyA->isHalfTy() && ElemTyB->isHalfTy() &&
          ElemTyC->isFloatTy() && M == 32 && N == 32 && K == 16) {
        Entry.IntrID = Intrinsic::amdgcn_mfma_f32_32x32x16_f16;
        Entry.ABVecTy = FixedVectorType::get(Type::getHalfTy(Ctx), 8);
        Entry.CDVecTy = FixedVectorType::get(Type::getFloatTy(Ctx), 16);
        Entry.IsMFMA = true;
        Entry.IsIU = false;
        return Entry;
      }
      // bf16 * bf16 + f32 -> f32, 16x16x32 and 32x32x16 (gfx942+)
      if (ST.hasGFX90AInsts() && ElemTyA->isBFloatTy() &&
          ElemTyB->isBFloatTy() && ElemTyC->isFloatTy()) {
        if (M == 16 && N == 16 && K == 32) {
          Entry.IntrID = Intrinsic::amdgcn_mfma_f32_16x16x32_bf16;
          Entry.ABVecTy = FixedVectorType::get(Type::getBFloatTy(Ctx), 8);
          Entry.CDVecTy = FixedVectorType::get(Type::getFloatTy(Ctx), 4);
          Entry.IsMFMA = true;
          Entry.IsIU = false;
          return Entry;
        }
        if (M == 32 && N == 32 && K == 16) {
          Entry.IntrID = Intrinsic::amdgcn_mfma_f32_32x32x16_bf16;
          Entry.ABVecTy = FixedVectorType::get(Type::getBFloatTy(Ctx), 8);
          Entry.CDVecTy = FixedVectorType::get(Type::getFloatTy(Ctx), 16);
          Entry.IsMFMA = true;
          Entry.IsIU = false;
          return Entry;
        }
      }
    }

    // --- f64 MFMA (gfx90a+) ---
    if (ST.hasGFX90AInsts() && ElemTyA->isDoubleTy() &&
        ElemTyB->isDoubleTy() && ElemTyC->isDoubleTy()) {
      // f64 * f64 + f64 -> f64, 16x16x4
      if (M == 16 && N == 16 && K == 4) {
        Entry.IntrID = Intrinsic::amdgcn_mfma_f64_16x16x4f64;
        // Intrinsic takes scalar f64 for A/B.
        Entry.ABVecTy = Type::getDoubleTy(Ctx);
        Entry.CDVecTy = FixedVectorType::get(Type::getDoubleTy(Ctx), 4);
        Entry.IsMFMA = true;
        Entry.IsIU = false;
        return Entry;
      }
      // f64 * f64 + f64 -> f64, 4x4x4
      if (M == 4 && N == 4 && K == 4) {
        Entry.IntrID = Intrinsic::amdgcn_mfma_f64_4x4x4f64;
        Entry.ABVecTy = Type::getDoubleTy(Ctx);
        Entry.CDVecTy = Type::getDoubleTy(Ctx);
        Entry.IsMFMA = true;
        Entry.IsIU = false;
        return Entry;
      }
    }
  }

  return std::nullopt;
}

/// Lower a coopmatrix.construct (scalar splat) call.
static Value *lowerConstruct(CallInst *CI, const GCNSubtarget &ST,
                             IRBuilder<> &Builder) {
  // Args: scalar_value(0), scope(1), rows(2), cols(3), use(4)
  Value *Scalar = CI->getArgOperand(0);

  Type *ElemTy;
  unsigned Scope, Rows, Cols, Use;
  if (!getCoopMatParams(CI->getType(), ElemTy, Scope, Rows, Cols, Use))
    report_fatal_error("coopmatrix.construct: invalid return type");

  FixedVectorType *VecTy = getConcreteVectorType(ST, ElemTy, Rows, Cols, Use);
  unsigned NumElems = VecTy->getNumElements();

  // If scalar type does not match vector element type, cast it.
  if (Scalar->getType() != VecTy->getElementType())
    Scalar = Builder.CreateBitOrPointerCast(Scalar, VecTy->getElementType());

  Value *Vec = PoisonValue::get(VecTy);
  for (unsigned I = 0; I < NumElems; ++I)
    Vec = Builder.CreateInsertElement(Vec, Scalar, I);

  return Vec;
}

/// Get the lane ID for the current work-item.
/// On wave32, this is just mbcnt_lo(-1, 0).
/// On wave64, this is mbcnt_lo(-1, 0) + mbcnt_hi(-1, mbcnt_lo_result).
static Value *getLaneId(const GCNSubtarget &ST, IRBuilder<> &Builder,
                        Module *M) {
  Value *AllOnes = ConstantInt::get(Builder.getInt32Ty(), ~0u);
  Value *Zero32 = ConstantInt::get(Builder.getInt32Ty(), 0);
  Function *MbcntLo =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_mbcnt_lo);
  Value *LaneId = Builder.CreateCall(MbcntLo, {AllOnes, Zero32});

  if (ST.getWavefrontSize() == 64) {
    Function *MbcntHi =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_mbcnt_hi);
    LaneId = Builder.CreateCall(MbcntHi, {AllOnes, LaneId});
  }

  return LaneId;
}

/// Compute (row, col) for each per-lane element of an MFMA matrix.
///
/// Verified against Triton's LinearLayoutConversions.cpp which cites ROCm's
/// amd_matrix_instruction_calculator. kWidth below is the per-lane element
/// count along the K dimension, which equals VecLen (the number of elements
/// in the concrete lane vector).
///
/// MFMA A operand (M × K):
///   row = lane % M
///   col = (lane / M) * kWidth + elem_idx
/// MFMA B operand (K × N):
///   row = (lane / N) * kWidth + elem_idx
///   col = lane % N
/// MFMA Accumulator 16×16 (4 elems/lane):
///   row = (lane / 16) * 4 + elem_idx
///   col = lane % 16
/// MFMA Accumulator 32×32 (16 elems/lane, 4 tiles × 4 elements):
///   row = (lane / 32) * 4 + tile * 8 + elem_idx
///   col = lane % 32
static void getMFMAElementCoords(
    IRBuilder<> &Builder, Value *LaneId, unsigned Rows, unsigned Cols,
    unsigned VecLen, unsigned Use,
    SmallVectorImpl<std::pair<Value *, Value *>> &Coords) {
  Value *LaneId64 = Builder.CreateZExt(LaneId, Builder.getInt64Ty());

  if (Use == MatrixA) {
    // MFMA A: row = lane % M, col = (lane / M) * kWidth + elem
    // kWidth is the per-lane element count along K, which equals VecLen
    // (the concrete vector type's element count). For 16x16x16 f16 this is
    // 4; for 32x32x8 f16 this is also 4 (8 K elems / 2 lane groups).
    unsigned M = Rows;
    unsigned kWidth = VecLen;
    Value *Row = Builder.CreateURem(LaneId64, Builder.getInt64(M));
    Value *LaneHi = Builder.CreateUDiv(LaneId64, Builder.getInt64(M));
    Value *BaseCol = Builder.CreateMul(LaneHi, Builder.getInt64(kWidth));
    for (unsigned E = 0; E < VecLen; ++E) {
      Value *Col = Builder.CreateAdd(BaseCol, Builder.getInt64(E));
      Coords.push_back({Row, Col});
    }
    return;
  }
  if (Use == MatrixB) {
    // MFMA B: col = lane % N, row = (lane / N) * kWidth + elem
    unsigned N = Cols;
    unsigned kWidth = VecLen;
    Value *Col = Builder.CreateURem(LaneId64, Builder.getInt64(N));
    Value *LaneHi = Builder.CreateUDiv(LaneId64, Builder.getInt64(N));
    Value *BaseRow = Builder.CreateMul(LaneHi, Builder.getInt64(kWidth));
    for (unsigned E = 0; E < VecLen; ++E) {
      Value *Row = Builder.CreateAdd(BaseRow, Builder.getInt64(E));
      Coords.push_back({Row, Col});
    }
    return;
  }

  // Accumulator (Use == Accumulator)
  if (Rows == 16 && Cols == 16) {
    // 4 elements per lane: row = (lane/16)*4 + elem, col = lane%16
    Value *BaseRow = Builder.CreateMul(
        Builder.CreateUDiv(LaneId64, Builder.getInt64(16)),
        Builder.getInt64(4));
    Value *Col = Builder.CreateURem(LaneId64, Builder.getInt64(16));
    for (unsigned E = 0; E < VecLen; ++E) {
      Value *Row = Builder.CreateAdd(BaseRow, Builder.getInt64(E));
      Coords.push_back({Row, Col});
    }
  } else if (Rows == 32 && Cols == 32) {
    // BUG-1 fix: 16 elements per lane arranged as 4 tiles of 4 rows each.
    // Half-wave stride is 4 (not 16) and tile stride is T*8 (not
    // (T>>1)*8 + (T&1)*4). Resulting rows for lane 0: {0..3, 8..11, 16..19,
    // 24..27}.
    Value *HalfWave = Builder.CreateLShr(LaneId64, Builder.getInt64(5));
    Value *HalfWaveOff = Builder.CreateMul(HalfWave, Builder.getInt64(4));
    Value *Col = Builder.CreateURem(LaneId64, Builder.getInt64(32));
    for (unsigned T = 0; T < 4; ++T) {
      Value *TileBase =
          Builder.CreateAdd(HalfWaveOff, Builder.getInt64(T * 8));
      for (unsigned E = 0; E < 4; ++E) {
        Value *Row = Builder.CreateAdd(TileBase, Builder.getInt64(E));
        Coords.push_back({Row, Col});
      }
    }
  } else {
    report_fatal_error("unsupported MFMA accumulator shape in "
                       "getMFMAElementCoords");
  }
}

/// Compute (row, col) for each per-lane element of a WMMA v2/v3 matrix
/// (gfx12+, wave32). 16x16 output: 32 lanes, 8 elements per lane.
/// Lower 16 lanes (0-15) cover rows 0-7, upper 16 lanes (16-31) cover rows
/// 8-15. Within each half-wave, elements are contiguous in the M dimension.
///   col = lane_id % 16
///   row = (lane_id / 16) * 8 + elem_idx
static void getWMMAv2ElementCoords(
    IRBuilder<> &Builder, Value *LaneId, unsigned VecLen,
    SmallVectorImpl<std::pair<Value *, Value *>> &Coords) {
  Value *LaneId64 = Builder.CreateZExt(LaneId, Builder.getInt64Ty());
  Value *Col = Builder.CreateURem(LaneId64, Builder.getInt64(16));
  Value *HalfWave = Builder.CreateUDiv(LaneId64, Builder.getInt64(16));
  Value *BaseRow = Builder.CreateMul(HalfWave, Builder.getInt64(8));
  for (unsigned E = 0; E < VecLen; ++E) {
    Value *Row = Builder.CreateAdd(BaseRow, Builder.getInt64(E));
    Coords.push_back({Row, Col});
  }
}

/// Compute (row, col) for each per-lane element of a WMMA v1 matrix
/// (gfx11, wave32). 16x16 output: 32 lanes, 8 elements per lane.
/// The v1 layout has stride-2 in the M dimension:
///   col = lane_id % 16
///   row = (lane_id / 16) + elem_idx * 2
/// This interleaves rows from the two half-waves: lane_hi=0 gets even rows
/// (0,2,4,6,8,10,12,14), lane_hi=1 gets odd rows (1,3,5,7,9,11,13,15).
/// Reference: Triton AMDWmmaEncodingAttr::getTileLayout version==1.
static void getWMMAv1ElementCoords(
    IRBuilder<> &Builder, Value *LaneId, unsigned VecLen,
    SmallVectorImpl<std::pair<Value *, Value *>> &Coords) {
  Value *LaneId64 = Builder.CreateZExt(LaneId, Builder.getInt64Ty());
  Value *Col = Builder.CreateURem(LaneId64, Builder.getInt64(16));
  Value *LaneHi = Builder.CreateUDiv(LaneId64, Builder.getInt64(16));
  for (unsigned E = 0; E < VecLen; ++E) {
    Value *Row =
        Builder.CreateAdd(LaneHi, Builder.getInt64((uint64_t)E * 2));
    Coords.push_back({Row, Col});
  }
}

/// Lower a coopmatrix.load call using HW-accurate lane layout.
/// Args: ptr(0), layout_imm(1), stride(2), scope(3), rows(4), cols(5), use(6)
/// Returns the loaded vector value.
static Value *lowerLoad(CallInst *CI, const GCNSubtarget &ST,
                        IRBuilder<> &Builder) {
  Value *Ptr = CI->getArgOperand(0);
  Value *Stride = CI->getArgOperand(2);

  auto *RetTy = dyn_cast<TargetExtType>(CI->getType());
  if (!RetTy)
    report_fatal_error("coopmatrix.load: return type is not a TargetExtType");

  Type *ElemTy;
  unsigned Scope, Rows, Cols, Use;
  if (!getCoopMatParams(RetTy, ElemTy, Scope, Rows, Cols, Use))
    report_fatal_error("coopmatrix.load: invalid cooperative matrix return "
                       "type");

  FixedVectorType *VecTy = getConcreteVectorType(ST, ElemTy, Rows, Cols, Use);
  unsigned VecLen = VecTy->getNumElements();
  unsigned ElemSize = ElemTy->getPrimitiveSizeInBits() / 8;

  Value *LaneId = getLaneId(ST, Builder, CI->getModule());
  Value *Stride64 = Builder.CreateZExt(Stride, Builder.getInt64Ty());

  // Compute per-element (row, col) based on HW layout.
  SmallVector<std::pair<Value *, Value *>, 16> Coords;
  if (ST.hasMAIInsts() && ST.isWave64()) {
    getMFMAElementCoords(Builder, LaneId, Rows, Cols, VecLen, Use, Coords);
  } else if (ST.hasWMMA128bInsts() && ST.isWave32()) {
    getWMMAv2ElementCoords(Builder, LaneId, VecLen, Coords);
  } else if (ST.hasWMMA256bInsts() && ST.isWave32()) {
    getWMMAv1ElementCoords(Builder, LaneId, VecLen, Coords);
  } else {
    report_fatal_error("coopmatrix.load: unsupported subtarget");
  }

  // Vectorized load: per-lane elements are contiguous in memory for all
  // supported MFMA/WMMA layouts. Instead of VecLen individual scalar loads,
  // compute the address of element 0 and emit one wide vector load.
  auto [Row0, Col0] = Coords[0];
  Value *ByteOff0 = Builder.CreateAdd(
      Builder.CreateMul(Row0,
                        Builder.CreateMul(Stride64,
                                          Builder.getInt64(ElemSize))),
      Builder.CreateMul(Col0, Builder.getInt64(ElemSize)));
  Value *BasePtr = Builder.CreateGEP(Builder.getInt8Ty(), Ptr, ByteOff0);
  Value *Vec = Builder.CreateAlignedLoad(VecTy, BasePtr,
                                         MaybeAlign(ElemSize));

  return Vec;
}

/// Lower a coopmatrix.store call using HW-accurate lane layout.
/// Args: matrix(0), ptr(1), layout_imm(2), stride(3), scope(4), rows(5),
///       cols(6), use(7)
static void lowerStore(CallInst *CI, Value *Matrix, const GCNSubtarget &ST,
                       IRBuilder<> &Builder) {
  Value *Ptr = CI->getArgOperand(1);
  Value *Stride = CI->getArgOperand(3);

  Type *MatTy = CI->getArgOperand(0)->getType();
  Function *Callee = CI->getCalledFunction();
  if (!isa<TargetExtType>(MatTy))
    MatTy = Callee->getFunctionType()->getParamType(0);

  Type *ElemTy;
  unsigned Scope, Rows, Cols, Use;
  if (!getCoopMatParams(MatTy, ElemTy, Scope, Rows, Cols, Use))
    report_fatal_error("coopmatrix.store: matrix operand is not a cooperative "
                       "matrix TargetExtType");

  FixedVectorType *VecTy = getConcreteVectorType(ST, ElemTy, Rows, Cols, Use);
  unsigned VecLen = VecTy->getNumElements();
  unsigned ElemSize = ElemTy->getPrimitiveSizeInBits() / 8;

  Value *LaneId = getLaneId(ST, Builder, CI->getModule());
  Value *Stride64 = Builder.CreateZExt(Stride, Builder.getInt64Ty());

  // Compute per-element (row, col) based on HW layout.
  SmallVector<std::pair<Value *, Value *>, 16> Coords;
  if (ST.hasMAIInsts() && ST.isWave64()) {
    getMFMAElementCoords(Builder, LaneId, Rows, Cols, VecLen, Use, Coords);
  } else if (ST.hasWMMA128bInsts() && ST.isWave32()) {
    getWMMAv2ElementCoords(Builder, LaneId, VecLen, Coords);
  } else if (ST.hasWMMA256bInsts() && ST.isWave32()) {
    getWMMAv1ElementCoords(Builder, LaneId, VecLen, Coords);
  } else {
    report_fatal_error("coopmatrix.store: unsupported subtarget");
  }

  // Vectorized store: per-lane elements are contiguous in memory.
  // Compute the address of element 0 and emit one wide vector store.
  auto [Row0, Col0] = Coords[0];
  Value *ByteOff0 = Builder.CreateAdd(
      Builder.CreateMul(Row0,
                        Builder.CreateMul(Stride64,
                                          Builder.getInt64(ElemSize))),
      Builder.CreateMul(Col0, Builder.getInt64(ElemSize)));
  Value *BasePtr = Builder.CreateGEP(Builder.getInt8Ty(), Ptr, ByteOff0);
  Builder.CreateAlignedStore(Matrix, BasePtr, MaybeAlign(ElemSize));
}

/// Lower a coopmatrix.length call to a constant.
//
// TODO(coopmatrix): coopmatrix_length does not take element type as an
// operand. For configs where element size affects per-lane vector length
// (e.g., FP8 MFMA which has 8 elems/lane vs f16's 4), this returns the
// wrong length. Fix requires extending the intrinsic signature.
// Phase 3a must avoid adding FP8 configs whose length diverges until
// this is resolved, OR emit a diagnoseUnsupported for length queries
// on affected configs. See BUG-7 in status.md §2.1.
static Value *lowerLength(CallInst *CI, const GCNSubtarget &ST) {
  // Args: scope(0), rows(1), cols(2), use(3)
  // We need the element type to determine the vector length, but length
  // has no TargetExtType return. We infer from the use-site context.
  // Actually, coopmatrix.length returns i32 and takes scalar args.
  // We need to figure out the element type. The intrinsic doesn't carry it
  // directly -- we need to look at the context or hardcode the mapping.
  //
  // For now, we use the rows/cols/use to determine the length. The element
  // type doesn't change the per-lane length for a given config on AMDGPU
  // (for the configs supported today; see BUG-7 above).

  auto *ScopeC = dyn_cast<ConstantInt>(CI->getArgOperand(0));
  auto *RowsC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  auto *ColsC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
  auto *UseC = dyn_cast<ConstantInt>(CI->getArgOperand(3));

  if (!ScopeC || !RowsC || !ColsC || !UseC)
    report_fatal_error("coopmatrix.length: non-constant arguments");

  unsigned Rows = RowsC->getZExtValue();
  unsigned Cols = ColsC->getZExtValue();
  unsigned Use = UseC->getZExtValue();

  unsigned Length = 0;

  if (ST.hasWMMA128bInsts() && ST.isWave32()) {
    // gfx12 w32: all matrix operands are 8 elements per lane
    Length = 8;
  } else if (ST.hasWMMA256bInsts() && ST.isWave32()) {
    // gfx11 w32 (WMMA v1): all matrix operands are 8 elements per lane
    Length = 8;
  } else if (ST.hasMAIInsts() && ST.isWave64()) {
    // MFMA: accumulator length depends on dimensions
    if (Use == Accumulator) {
      if (Rows == 16 && Cols == 16)
        Length = 4;
      else if (Rows == 32 && Cols == 32)
        Length = 16;
    } else {
      // A/B operands
      Length = 4;
    }
  }

  if (Length == 0)
    report_fatal_error("Unsupported cooperative matrix length configuration");

  return ConstantInt::get(CI->getType(), Length);
}

/// Lower a coopmatrix.muladd call.
/// \p A, \p B, \p C are the already-lowered concrete vector values for the
/// matrix operands. The original TargetExtType information is extracted from
/// the call's formal parameter types (which still carry the TargetExtType).
static Value *lowerMulAdd(CallInst *CI, Value *A, Value *B, Value *C,
                          const GCNSubtarget &ST, IRBuilder<> &Builder) {
  // Args: matA(0), matB(1), matC(2), operands_imm(3), scope(4), M(5), N(6),
  // K(7), typeInterpA_imm(8), typeInterpB_imm(9), typeInterpC_imm(10)
  auto *MC = dyn_cast<ConstantInt>(CI->getArgOperand(5));
  auto *NC = dyn_cast<ConstantInt>(CI->getArgOperand(6));
  auto *KC = dyn_cast<ConstantInt>(CI->getArgOperand(7));
  if (!MC || !NC || !KC)
    report_fatal_error("coopmatrix.muladd: non-constant M/N/K");

  unsigned M = MC->getZExtValue();
  unsigned N = NC->getZExtValue();
  unsigned K = KC->getZExtValue();

  // Get element types from the original TargetExtType of the operands.
  // The call instruction's formal parameter types still have the TargetExtType
  // even though the actual argument values may have been replaced already.
  Type *OrigTyA = CI->getArgOperand(0)->getType();
  Type *OrigTyB = CI->getArgOperand(1)->getType();
  Type *OrigTyC = CI->getArgOperand(2)->getType();

  // If the operands have already been replaced (no longer TargetExtType),
  // we need the original types. We store them from the intrinsic's
  // function type.
  Function *Callee = CI->getCalledFunction();
  FunctionType *FTy = Callee->getFunctionType();
  if (!isa<TargetExtType>(OrigTyA))
    OrigTyA = FTy->getParamType(0);
  if (!isa<TargetExtType>(OrigTyB))
    OrigTyB = FTy->getParamType(1);
  if (!isa<TargetExtType>(OrigTyC))
    OrigTyC = FTy->getParamType(2);

  Type *ElemTyA, *ElemTyB, *ElemTyC;
  unsigned ScopeA, RowsA, ColsA, UseA;
  unsigned ScopeB, RowsB, ColsB, UseB;
  unsigned ScopeC, RowsC, ColsC, UseC;
  if (!getCoopMatParams(OrigTyA, ElemTyA, ScopeA, RowsA, ColsA, UseA) ||
      !getCoopMatParams(OrigTyB, ElemTyB, ScopeB, RowsB, ColsB, UseB) ||
      !getCoopMatParams(OrigTyC, ElemTyC, ScopeC, RowsC, ColsC, UseC))
    report_fatal_error("coopmatrix.muladd: operand is not a cooperative "
                       "matrix TargetExtType");

  LLVMContext &Ctx = CI->getContext();
  auto Entry = lookupMulAdd(ST, Ctx, ElemTyA, ElemTyB, ElemTyC, M, N, K);
  if (!Entry)
    report_fatal_error("Unsupported cooperative matrix muladd configuration "
                       "for AMDGPU target");

  // Bitcast A/B to the intrinsic's expected type if needed.
  // For example, i8 cooperative matrices are represented as <8 x i8> but
  // the WMMA IU intrinsic expects <2 x i32>.
  if (A->getType() != Entry->ABVecTy)
    A = Builder.CreateBitCast(A, Entry->ABVecTy);
  if (B->getType() != Entry->ABVecTy)
    B = Builder.CreateBitCast(B, Entry->ABVecTy);
  if (C->getType() != Entry->CDVecTy)
    C = Builder.CreateBitCast(C, Entry->CDVecTy);

  Value *Result;
  if (Entry->IsMFMA) {
    // MFMA: (A, B, C, cbsz=0, abid=0, blgp=0) -> D
    // Not overloaded -- concrete types.
    Result = Builder.CreateIntrinsic(
        Entry->IntrID, {},
        {A, B, C, Builder.getInt32(0), Builder.getInt32(0),
         Builder.getInt32(0)});
  } else if (Entry->IsIU) {
    // WMMA IU: (A_sign=false, A, B_sign=false, B, C, clamp=false) -> D
    // Overloaded: type params are [CDVecTy, ABVecTy]
    Result = Builder.CreateIntrinsic(
        Entry->IntrID, {Entry->CDVecTy, Entry->ABVecTy},
        {Builder.getFalse(), A, Builder.getFalse(), B, C, Builder.getFalse()});
  } else {
    // WMMA: (A, B, C) -> D
    // Overloaded: type params are [CDVecTy, ABVecTy]
    Result = Builder.CreateIntrinsic(Entry->IntrID,
                                     {Entry->CDVecTy, Entry->ABVecTy},
                                     {A, B, C});
  }

  return Result;
}

/// Translate a coopmatrix_muladd_scaled format enum (0=FP8_E5M2,
/// 1=FP8_E4M3, 2=BF8, 3=FP6_E3M2, 4=FP6_E2M3, 5=FP4_E2M1, 6=I8) to
/// APFloatBase::Semantics. Returns std::nullopt for I8 (no APFloat
/// equivalent) or invalid values. See BUG-8 in status.md §2.1.
[[maybe_unused]] static std::optional<APFloatBase::Semantics>
coopmatrixScaledFormatToAPFloat(unsigned Fmt) {
  switch (Fmt) {
  case 0:
    return APFloatBase::S_Float8E5M2;
  case 1:
    return APFloatBase::S_Float8E4M3FN;
  case 2:
    return APFloatBase::S_Float8E5M2; // "BF8" treated as E5M2
  case 3:
    return APFloatBase::S_Float6E3M2FN;
  case 4:
    return APFloatBase::S_Float6E2M3FN;
  case 5:
    return APFloatBase::S_Float4E2M1FN;
  case 6:
    return std::nullopt; // I8 — no APFloat equiv
  default:
    return std::nullopt;
  }
}

/// Unary operations on cooperative matrix element type.
enum CoopMatUnaryOp : unsigned {
  UnaryNegate = 0,
  UnaryAbs = 1,
  UnaryNot = 2,
  UnaryCeil = 3,
  UnaryFloor = 4,
  UnaryRound = 5,
  UnaryTrunc = 6,
};

/// Binary operations on cooperative matrix element type.
enum CoopMatBinaryOp : unsigned {
  BinaryAdd = 0,
  BinarySub = 1,
  BinaryMul = 2,
  BinaryDiv = 3,
  BinaryFRem = 4,
  BinaryAnd = 5,
  BinaryOr = 6,
  BinaryXor = 7,
  BinaryShl = 8,
  BinaryAShr = 9,
  BinaryLShr = 10,
  BinaryMin = 11,
  BinaryMax = 12,
};

/// Lower a coopmatrix.unary call on the concrete per-lane vector.
/// op(imm): 0=Negate, 1=Abs, 2=Not, 3=Ceil, 4=Floor, 5=Round, 6=Trunc
/// Args: matrix(0), op(1), scope(2), rows(3), cols(4), use(5)
static Value *lowerUnary(CallInst *CI, Value *Mat, IRBuilder<> &Builder) {
  auto *OpC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!OpC)
    report_fatal_error("coopmatrix.unary: non-constant op");
  unsigned Op = OpC->getZExtValue();

  auto *VecTy = cast<FixedVectorType>(Mat->getType());
  Type *ElemTy = VecTy->getElementType();

  switch (Op) {
  case UnaryNegate:
    if (ElemTy->isFloatingPointTy())
      return Builder.CreateFNeg(Mat);
    return Builder.CreateNeg(Mat);
  case UnaryAbs: {
    if (ElemTy->isFloatingPointTy()) {
      Function *FAbs = Intrinsic::getOrInsertDeclaration(
          CI->getModule(), Intrinsic::fabs, {VecTy});
      return Builder.CreateCall(FAbs, {Mat});
    }
    Function *IAbs = Intrinsic::getOrInsertDeclaration(
        CI->getModule(), Intrinsic::abs, {VecTy});
    return Builder.CreateCall(IAbs, {Mat, Builder.getFalse()});
  }
  case UnaryNot:
    return Builder.CreateNot(Mat);
  case UnaryCeil: {
    Function *F = Intrinsic::getOrInsertDeclaration(
        CI->getModule(), Intrinsic::ceil, {VecTy});
    return Builder.CreateCall(F, {Mat});
  }
  case UnaryFloor: {
    Function *F = Intrinsic::getOrInsertDeclaration(
        CI->getModule(), Intrinsic::floor, {VecTy});
    return Builder.CreateCall(F, {Mat});
  }
  case UnaryRound: {
    Function *F = Intrinsic::getOrInsertDeclaration(
        CI->getModule(), Intrinsic::round, {VecTy});
    return Builder.CreateCall(F, {Mat});
  }
  case UnaryTrunc: {
    Function *F = Intrinsic::getOrInsertDeclaration(
        CI->getModule(), Intrinsic::trunc, {VecTy});
    return Builder.CreateCall(F, {Mat});
  }
  default:
    report_fatal_error("coopmatrix.unary: unknown op " + Twine(Op));
  }
}

/// Lower a coopmatrix.binary call on the concrete per-lane vector.
/// op(imm): 0=Add, 1=Sub, 2=Mul, 3=Div, 4=FRem, 5=And, 6=Or, 7=Xor,
///          8=Shl, 9=AShr, 10=LShr, 11=Min, 12=Max
/// Args: matA(0), matB(1), op(2), scope(3), rows(4), cols(5), use(6)
static Value *lowerBinary(CallInst *CI, Value *A, Value *B,
                          IRBuilder<> &Builder) {
  auto *OpC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
  if (!OpC)
    report_fatal_error("coopmatrix.binary: non-constant op");
  unsigned Op = OpC->getZExtValue();

  auto *VecTy = cast<FixedVectorType>(A->getType());
  Type *ElemTy = VecTy->getElementType();
  bool IsFP = ElemTy->isFloatingPointTy();

  switch (Op) {
  case BinaryAdd:
    return IsFP ? Builder.CreateFAdd(A, B) : Builder.CreateAdd(A, B);
  case BinarySub:
    return IsFP ? Builder.CreateFSub(A, B) : Builder.CreateSub(A, B);
  case BinaryMul:
    return IsFP ? Builder.CreateFMul(A, B) : Builder.CreateMul(A, B);
  case BinaryDiv:
    return IsFP ? Builder.CreateFDiv(A, B) : Builder.CreateSDiv(A, B);
  case BinaryFRem:
    return Builder.CreateFRem(A, B);
  case BinaryAnd:
    return Builder.CreateAnd(A, B);
  case BinaryOr:
    return Builder.CreateOr(A, B);
  case BinaryXor:
    return Builder.CreateXor(A, B);
  case BinaryShl:
    return Builder.CreateShl(A, B);
  case BinaryAShr:
    return Builder.CreateAShr(A, B);
  case BinaryLShr:
    return Builder.CreateLShr(A, B);
  case BinaryMin:
    if (IsFP) {
      Function *F = Intrinsic::getOrInsertDeclaration(
          CI->getModule(), Intrinsic::minnum, {VecTy});
      return Builder.CreateCall(F, {A, B});
    }
    return Builder.CreateSelect(Builder.CreateICmpSLT(A, B), A, B);
  case BinaryMax:
    if (IsFP) {
      Function *F = Intrinsic::getOrInsertDeclaration(
          CI->getModule(), Intrinsic::maxnum, {VecTy});
      return Builder.CreateCall(F, {A, B});
    }
    return Builder.CreateSelect(Builder.CreateICmpSGT(A, B), A, B);
  default:
    report_fatal_error("coopmatrix.binary: unknown op " + Twine(Op));
  }
}

/// Lower a coopmatrix.convert call.
/// Args: src_matrix(0), scope(1), rows(2), cols(3), src_use(4), dst_use(5)
static Value *lowerConvert(CallInst *CI, Value *Src, const GCNSubtarget &ST,
                           IRBuilder<> &Builder) {
  auto *DstTET = dyn_cast<TargetExtType>(CI->getType());
  if (!DstTET)
    report_fatal_error("coopmatrix.convert: invalid return type");

  Type *DstElemTy;
  unsigned Scope, Rows, Cols, DstUse;
  if (!getCoopMatParams(DstTET, DstElemTy, Scope, Rows, Cols, DstUse))
    report_fatal_error("coopmatrix.convert: invalid return type");

  FixedVectorType *DstVecTy =
      getConcreteVectorType(ST, DstElemTy, Rows, Cols, DstUse);
  Type *SrcElemTy = cast<FixedVectorType>(Src->getType())->getElementType();

  if (SrcElemTy == DstVecTy->getElementType())
    return Src;

  bool SrcFP = SrcElemTy->isFloatingPointTy();
  bool DstFP = DstElemTy->isFloatingPointTy();

  if (SrcFP && DstFP) {
    unsigned SrcBits = SrcElemTy->getPrimitiveSizeInBits();
    unsigned DstBits = DstElemTy->getPrimitiveSizeInBits();
    if (DstBits > SrcBits)
      return Builder.CreateFPExt(Src, DstVecTy);
    return Builder.CreateFPTrunc(Src, DstVecTy);
  }
  if (!SrcFP && !DstFP)
    return Builder.CreateIntCast(Src, DstVecTy,
                                 SrcElemTy->isIntegerTy() /* signed */);
  if (SrcFP && !DstFP)
    return Builder.CreateFPToSI(Src, DstVecTy);
  // !SrcFP && DstFP
  return Builder.CreateSIToFP(Src, DstVecTy);
}

/// Lower a coopmatrix.extract call.
/// Args: matrix(0), index(1), scope(2), rows(3), cols(4), use(5)
static Value *lowerExtract(CallInst *CI, Value *Mat, IRBuilder<> &Builder) {
  Value *Idx = CI->getArgOperand(1);
  return Builder.CreateExtractElement(Mat, Idx);
}

/// Lower a coopmatrix.insert call.
/// Args: matrix(0), value(1), index(2), scope(3), rows(4), cols(5), use(6)
static Value *lowerInsert(CallInst *CI, Value *Mat, Value *Val,
                          IRBuilder<> &Builder) {
  Value *Idx = CI->getArgOperand(2);
  return Builder.CreateInsertElement(Mat, Val, Idx);
}

/// Lower a coopmatrix.get_coord call.
/// Returns <2 x i32> = {row, col} for the given per-lane element index.
/// Args: matrix(0), index(1), scope(2), rows(3), cols(4), use(5)
static Value *lowerGetCoord(CallInst *CI, const GCNSubtarget &ST,
                            IRBuilder<> &Builder) {
  // Get the matrix params from the type of the matrix operand.
  Function *Callee = CI->getCalledFunction();
  Type *MatTy = Callee->getFunctionType()->getParamType(0);
  Type *ElemTy;
  unsigned Scope, Rows, Cols, Use;
  if (!getCoopMatParams(MatTy, ElemTy, Scope, Rows, Cols, Use))
    report_fatal_error("coopmatrix.get_coord: invalid matrix type");

  FixedVectorType *VecTy = getConcreteVectorType(ST, ElemTy, Rows, Cols, Use);
  unsigned VecLen = VecTy->getNumElements();

  Value *LaneId = getLaneId(ST, Builder, CI->getModule());
  Value *Idx = CI->getArgOperand(1);

  SmallVector<std::pair<Value *, Value *>, 16> Coords;
  if (ST.hasMAIInsts() && ST.isWave64()) {
    getMFMAElementCoords(Builder, LaneId, Rows, Cols, VecLen, Use, Coords);
  } else if (ST.hasWMMA128bInsts() && ST.isWave32()) {
    getWMMAv2ElementCoords(Builder, LaneId, VecLen, Coords);
  } else if (ST.hasWMMA256bInsts() && ST.isWave32()) {
    getWMMAv1ElementCoords(Builder, LaneId, VecLen, Coords);
  } else {
    report_fatal_error("coopmatrix.get_coord: unsupported subtarget");
  }

  // For constant index, pick directly. For dynamic, build a select chain.
  if (auto *IdxC = dyn_cast<ConstantInt>(Idx)) {
    unsigned E = IdxC->getZExtValue();
    if (E >= Coords.size())
      report_fatal_error("coopmatrix.get_coord: index out of range");
    auto [Row, Col] = Coords[E];
    Value *Result = PoisonValue::get(FixedVectorType::get(Builder.getInt32Ty(), 2));
    Result = Builder.CreateInsertElement(
        Result, Builder.CreateTrunc(Row, Builder.getInt32Ty()), (uint64_t)0);
    Result = Builder.CreateInsertElement(
        Result, Builder.CreateTrunc(Col, Builder.getInt32Ty()), (uint64_t)1);
    return Result;
  }

  // Dynamic index: select chain over all possible values.
  Value *RowResult = Builder.getInt64(0);
  Value *ColResult = Builder.getInt64(0);
  Value *Idx64 = Builder.CreateZExt(Idx, Builder.getInt64Ty());
  for (unsigned E = 0; E < Coords.size(); ++E) {
    auto [R, C] = Coords[E];
    Value *Cmp = Builder.CreateICmpEQ(Idx64, Builder.getInt64(E));
    RowResult = Builder.CreateSelect(Cmp, R, RowResult);
    ColResult = Builder.CreateSelect(Cmp, C, ColResult);
  }
  Value *Result = PoisonValue::get(FixedVectorType::get(Builder.getInt32Ty(), 2));
  Result = Builder.CreateInsertElement(
      Result, Builder.CreateTrunc(RowResult, Builder.getInt32Ty()), (uint64_t)0);
  Result = Builder.CreateInsertElement(
      Result, Builder.CreateTrunc(ColResult, Builder.getInt32Ty()), (uint64_t)1);
  return Result;
}

/// Lower a coopmatrix.prefetch call.
/// Args: ptr(0), rows(1), cols(2), cache_level(3), layout(4), stride(5),
///       scope(6)
static void lowerPrefetch(CallInst *CI, IRBuilder<> &Builder) {
  Value *Ptr = CI->getArgOperand(0);
  // Emit a generic prefetch. On AMDGPU this will be lowered to
  // s_prefetch_data or similar by the backend.
  Function *PrefetchFn = Intrinsic::getOrInsertDeclaration(
      CI->getModule(), Intrinsic::prefetch, {Ptr->getType()});
  // prefetch(ptr, rw=0(read), locality=3, cache_type=1(data))
  Builder.CreateCall(PrefetchFn,
                     {Ptr, Builder.getInt32(0), Builder.getInt32(3),
                      Builder.getInt32(1)});
}

/// Lower a coopmatrix.muladd_ext call.
/// D = op(A) * op(B) + op(C) with modifiers.
/// Args: matA(0), matB(1), matC(2), operands(3), modifiers(4), scope(5),
///       M(6), N(7), K(8)
static Value *lowerMulAddExt(CallInst *CI, Value *A, Value *B, Value *C,
                             const GCNSubtarget &ST, IRBuilder<> &Builder) {
  auto *ModC = dyn_cast<ConstantInt>(CI->getArgOperand(4));
  if (!ModC)
    report_fatal_error("coopmatrix.muladd_ext: non-constant modifiers");
  unsigned Mods = ModC->getZExtValue();

  // Apply pre-operation modifiers on the concrete vectors.
  // bit 0: neg_A, bit 1: neg_B, bit 2: neg_C, bit 3: abs_C
  if (Mods & 1)
    A = A->getType()->isFPOrFPVectorTy() ? Builder.CreateFNeg(A)
                                          : Builder.CreateNeg(A);
  if (Mods & 2)
    B = B->getType()->isFPOrFPVectorTy() ? Builder.CreateFNeg(B)
                                          : Builder.CreateNeg(B);
  if (Mods & 4)
    C = C->getType()->isFPOrFPVectorTy() ? Builder.CreateFNeg(C)
                                          : Builder.CreateNeg(C);
  if (Mods & 8) {
    auto *VecTy = cast<VectorType>(C->getType());
    Function *FAbs = Intrinsic::getOrInsertDeclaration(
        CI->getModule(), Intrinsic::fabs, {VecTy});
    C = Builder.CreateCall(FAbs, {C});
  }

  // Delegate to regular muladd lowering.
  Value *Result = lowerMulAdd(CI, A, B, C, ST, Builder);

  // bit 4: clamp — saturate result to [0, max_float]
  if (Mods & 16) {
    auto *VecTy = cast<VectorType>(Result->getType());
    if (VecTy->getElementType()->isFloatingPointTy()) {
      Value *Zero = ConstantFP::get(VecTy, 0.0);
      Function *MaxFn = Intrinsic::getOrInsertDeclaration(
          CI->getModule(), Intrinsic::maxnum, {VecTy});
      Result = Builder.CreateCall(MaxFn, {Result, Zero});
    }
  }

  return Result;
}

/// Lower a coopmatrix.load_checked call via masked vector load.
/// Args: ptr(0), x_off(1), y_off(2), height(3), width(4), layout(5),
///       stride(6), scope(7), rows(8), cols(9), use(10)
static Value *lowerLoadChecked(CallInst *CI, const GCNSubtarget &ST,
                               IRBuilder<> &Builder) {
  Value *Ptr = CI->getArgOperand(0);
  Value *XOff = CI->getArgOperand(1);
  Value *YOff = CI->getArgOperand(2);
  Value *Height = CI->getArgOperand(3);
  Value *Width = CI->getArgOperand(4);
  Value *Stride = CI->getArgOperand(6);

  auto *RetTy = dyn_cast<TargetExtType>(CI->getType());
  if (!RetTy)
    report_fatal_error("coopmatrix.load_checked: invalid return type");

  Type *ElemTy;
  unsigned Scope, Rows, Cols, Use;
  if (!getCoopMatParams(RetTy, ElemTy, Scope, Rows, Cols, Use))
    report_fatal_error("coopmatrix.load_checked: invalid return type");

  FixedVectorType *VecTy = getConcreteVectorType(ST, ElemTy, Rows, Cols, Use);
  unsigned VecLen = VecTy->getNumElements();
  unsigned ElemSize = ElemTy->getPrimitiveSizeInBits() / 8;

  Value *LaneId = getLaneId(ST, Builder, CI->getModule());
  Value *Stride64 = Builder.CreateZExt(Stride, Builder.getInt64Ty());

  SmallVector<std::pair<Value *, Value *>, 16> Coords;
  if (ST.hasMAIInsts() && ST.isWave64())
    getMFMAElementCoords(Builder, LaneId, Rows, Cols, VecLen, Use, Coords);
  else if (ST.hasWMMA128bInsts() && ST.isWave32())
    getWMMAv2ElementCoords(Builder, LaneId, VecLen, Coords);
  else if (ST.hasWMMA256bInsts() && ST.isWave32())
    getWMMAv1ElementCoords(Builder, LaneId, VecLen, Coords);
  else
    report_fatal_error("coopmatrix.load_checked: unsupported subtarget");

  // Bounds-checked per-element load. Out-of-bounds elements get zero.
  Value *XOff64 = Builder.CreateZExt(XOff, Builder.getInt64Ty());
  Value *YOff64 = Builder.CreateZExt(YOff, Builder.getInt64Ty());
  Value *Height64 = Builder.CreateZExt(Height, Builder.getInt64Ty());
  Value *Width64 = Builder.CreateZExt(Width, Builder.getInt64Ty());

  Value *Vec = ConstantAggregateZero::get(VecTy);
  for (unsigned E = 0; E < VecLen; ++E) {
    auto [Row, Col] = Coords[E];
    Value *AbsRow = Builder.CreateAdd(Row, YOff64);
    Value *AbsCol = Builder.CreateAdd(Col, XOff64);
    Value *InBounds = Builder.CreateAnd(
        Builder.CreateICmpULT(AbsRow, Height64),
        Builder.CreateICmpULT(AbsCol, Width64));
    Value *ByteOff = Builder.CreateAdd(
        Builder.CreateMul(
            Row, Builder.CreateMul(Stride64, Builder.getInt64(ElemSize))),
        Builder.CreateMul(Col, Builder.getInt64(ElemSize)));
    Value *ElemPtr = Builder.CreateGEP(Builder.getInt8Ty(), Ptr, ByteOff);
    Value *Elem = Builder.CreateAlignedLoad(ElemTy, ElemPtr,
                                            MaybeAlign(ElemSize));
    Value *Zero = Constant::getNullValue(ElemTy);
    Value *Val = Builder.CreateSelect(InBounds, Elem, Zero);
    Vec = Builder.CreateInsertElement(Vec, Val, E);
  }
  return Vec;
}

/// Lower a coopmatrix.store_checked call via masked per-element store.
/// Args: matrix(0), ptr(1), x_off(2), y_off(3), height(4), width(5),
///       layout(6), stride(7), scope(8), rows(9), cols(10), use(11)
static void lowerStoreChecked(CallInst *CI, Value *Matrix,
                              const GCNSubtarget &ST, IRBuilder<> &Builder) {
  Value *Ptr = CI->getArgOperand(1);
  Value *XOff = CI->getArgOperand(2);
  Value *YOff = CI->getArgOperand(3);
  Value *Height = CI->getArgOperand(4);
  Value *Width = CI->getArgOperand(5);
  Value *Stride = CI->getArgOperand(7);

  Type *MatTy = CI->getArgOperand(0)->getType();
  Function *Callee = CI->getCalledFunction();
  if (!isa<TargetExtType>(MatTy))
    MatTy = Callee->getFunctionType()->getParamType(0);

  Type *ElemTy;
  unsigned Scope, Rows, Cols, Use;
  if (!getCoopMatParams(MatTy, ElemTy, Scope, Rows, Cols, Use))
    report_fatal_error("coopmatrix.store_checked: invalid matrix type");

  FixedVectorType *VecTy = getConcreteVectorType(ST, ElemTy, Rows, Cols, Use);
  unsigned VecLen = VecTy->getNumElements();
  unsigned ElemSize = ElemTy->getPrimitiveSizeInBits() / 8;

  Value *LaneId = getLaneId(ST, Builder, CI->getModule());
  Value *Stride64 = Builder.CreateZExt(Stride, Builder.getInt64Ty());

  SmallVector<std::pair<Value *, Value *>, 16> Coords;
  if (ST.hasMAIInsts() && ST.isWave64())
    getMFMAElementCoords(Builder, LaneId, Rows, Cols, VecLen, Use, Coords);
  else if (ST.hasWMMA128bInsts() && ST.isWave32())
    getWMMAv2ElementCoords(Builder, LaneId, VecLen, Coords);
  else if (ST.hasWMMA256bInsts() && ST.isWave32())
    getWMMAv1ElementCoords(Builder, LaneId, VecLen, Coords);
  else
    report_fatal_error("coopmatrix.store_checked: unsupported subtarget");

  Value *XOff64 = Builder.CreateZExt(XOff, Builder.getInt64Ty());
  Value *YOff64 = Builder.CreateZExt(YOff, Builder.getInt64Ty());
  Value *Height64 = Builder.CreateZExt(Height, Builder.getInt64Ty());
  Value *Width64 = Builder.CreateZExt(Width, Builder.getInt64Ty());

  for (unsigned E = 0; E < VecLen; ++E) {
    auto [Row, Col] = Coords[E];
    Value *AbsRow = Builder.CreateAdd(Row, YOff64);
    Value *AbsCol = Builder.CreateAdd(Col, XOff64);
    Value *InBounds = Builder.CreateAnd(
        Builder.CreateICmpULT(AbsRow, Height64),
        Builder.CreateICmpULT(AbsCol, Width64));
    Value *ByteOff = Builder.CreateAdd(
        Builder.CreateMul(
            Row, Builder.CreateMul(Stride64, Builder.getInt64(ElemSize))),
        Builder.CreateMul(Col, Builder.getInt64(ElemSize)));
    Value *ElemPtr = Builder.CreateGEP(Builder.getInt8Ty(), Ptr, ByteOff);
    Value *Elem = Builder.CreateExtractElement(Matrix, E);

    // Masked store: use llvm.masked.store to conditionally skip OOB writes.
    // Create a single-element vector for the masked store.
    auto *ScalarVecTy = FixedVectorType::get(ElemTy, 1);
    Value *ScalarVec = Builder.CreateInsertElement(
        PoisonValue::get(ScalarVecTy), Elem, (uint64_t)0);
    auto *MaskTy = FixedVectorType::get(Builder.getInt1Ty(), 1);
    Value *Mask = Builder.CreateInsertElement(
        PoisonValue::get(MaskTy), InBounds, (uint64_t)0);
    Builder.CreateMaskedStore(ScalarVec, ElemPtr, Align(ElemSize), Mask);
  }
}

/// The main pass implementation.
class AMDGPULowerCooperativeMatrix : public ModulePass {
public:
  static char ID;

  AMDGPULowerCooperativeMatrix() : ModulePass(ID) {}

  bool run(Module &M, const TargetMachine &TM);
  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // namespace

/// Check if a module contains any cooperative matrix intrinsic calls.
static bool hasCoopMatrixIntrinsics(const Module &M) {
  for (const Function &F : M) {
    if (!F.isDeclaration())
      continue;
    if (F.getName().starts_with("llvm.coopmatrix."))
      return true;
  }
  return false;
}

/// Process a single function: rewrite all cooperative matrix intrinsic calls.
static bool processFunction(Function &F, const GCNSubtarget &ST) {
  bool Changed = false;

  // Collect all coopmatrix calls first, then process them.
  // We need a two-phase approach:
  //  1. Collect all calls
  //  2. Process in order (operands of later calls may reference results of
  //     earlier calls that have already been lowered)
  //
  // We use a DenseMap to track TargetExtType Value -> concrete vector Value
  // replacements.
  DenseMap<Value *, Value *> ValueMap;
  SmallVector<CallInst *, 16> CoopCalls;

  for (BasicBlock &BB : F)
    for (Instruction &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      Function *Callee = CI->getCalledFunction();
      if (!Callee)
        continue;
      auto IID = Callee->getIntrinsicID();
      switch (IID) {
      // Set 1: baseline (implemented in this pass).
      case Intrinsic::coopmatrix_muladd:
      case Intrinsic::coopmatrix_construct:
      case Intrinsic::coopmatrix_length:
      case Intrinsic::coopmatrix_load:
      case Intrinsic::coopmatrix_store:
      // Set 2: extended element-wise / metadata ops (stubbed in Phase 2).
      case Intrinsic::coopmatrix_unary:
      case Intrinsic::coopmatrix_binary:
      case Intrinsic::coopmatrix_convert:
      case Intrinsic::coopmatrix_reduce:
      case Intrinsic::coopmatrix_extract:
      case Intrinsic::coopmatrix_insert:
      case Intrinsic::coopmatrix_get_coord:
      case Intrinsic::coopmatrix_prefetch:
      // Set 3: performance / sparse / checked ops (stubbed in Phase 2).
      case Intrinsic::coopmatrix_muladd_ext:
      case Intrinsic::coopmatrix_muladd_sparse:
      case Intrinsic::coopmatrix_muladd_scaled:
      case Intrinsic::coopmatrix_load_checked:
      case Intrinsic::coopmatrix_store_checked:
        CoopCalls.push_back(CI);
        break;
      default:
        break;
      }
    }

  if (CoopCalls.empty())
    return false;

  // BUG-4 guard: if any coopmatrix call has a use that is not itself a
  // coopmatrix intrinsic (or a PHI of a coopmatrix type — handled below),
  // the erase loop at the end cannot safely replace the TargetExtType
  // value. The previous implementation RAUW'd such uses to
  // UndefValue::get(TargetExtType) which then crashed SelectionDAG.
  //
  // Today no Phase 2 test exercises this case (all tests form
  // load/construct -> muladd -> store chains). Refuse it explicitly so
  // we never silently miscompile. Phase 3 will expand the accepted graph.
  auto isCoopCall = [](const Value *V) {
    const auto *CI = dyn_cast<CallInst>(V);
    if (!CI)
      return false;
    const Function *Callee = CI->getCalledFunction();
    if (!Callee)
      return false;
    switch (Callee->getIntrinsicID()) {
    case Intrinsic::coopmatrix_muladd:
    case Intrinsic::coopmatrix_construct:
    case Intrinsic::coopmatrix_length:
    case Intrinsic::coopmatrix_load:
    case Intrinsic::coopmatrix_store:
    case Intrinsic::coopmatrix_unary:
    case Intrinsic::coopmatrix_binary:
    case Intrinsic::coopmatrix_convert:
    case Intrinsic::coopmatrix_reduce:
    case Intrinsic::coopmatrix_extract:
    case Intrinsic::coopmatrix_insert:
    case Intrinsic::coopmatrix_get_coord:
    case Intrinsic::coopmatrix_prefetch:
    case Intrinsic::coopmatrix_muladd_ext:
    case Intrinsic::coopmatrix_muladd_sparse:
    case Intrinsic::coopmatrix_muladd_scaled:
    case Intrinsic::coopmatrix_load_checked:
    case Intrinsic::coopmatrix_store_checked:
      return true;
    default:
      return false;
    }
  };
  for (CallInst *CI : CoopCalls) {
    // Only coopmatrix-typed producers matter (length returns i32 and is
    // allowed to be used by anything).
    if (!isa<TargetExtType>(CI->getType()))
      continue;
    for (const Use &U : CI->uses()) {
      const User *Usr = U.getUser();
      if (isCoopCall(Usr))
        continue;
      if (isa<PHINode>(Usr))
        continue;
      report_fatal_error("coopmatrix value with non-coopmatrix use (e.g., "
                         "return or store of coop matrix) is not yet "
                         "supported in AMDGPULowerCooperativeMatrix");
    }
  }

  // Phase 0: Rewrite PHI nodes that have cooperative matrix TargetExtType.
  // Create new PHIs with the concrete vector type, and map old→new.
  SmallVector<PHINode *, 4> CoopPhis;
  for (auto &BB : F)
    for (auto &Phi : BB.phis())
      if (isa<TargetExtType>(Phi.getType()) &&
          cast<TargetExtType>(Phi.getType())
              ->getName()
              .starts_with("spirv.CooperativeMatrixKHR"))
        CoopPhis.push_back(&Phi);

  for (PHINode *Phi : CoopPhis) {
    auto *TET = cast<TargetExtType>(Phi->getType());
    Type *ElemTy;
    unsigned Scope, Rows, Cols, Use;
    if (!getCoopMatParams(TET, ElemTy, Scope, Rows, Cols, Use))
      continue;
    FixedVectorType *VecTy = getConcreteVectorType(ST, ElemTy, Rows, Cols, Use);
    PHINode *NewPhi =
        PHINode::Create(VecTy, Phi->getNumIncomingValues(), "", Phi->getIterator());
    // Incoming values will be patched after all coopmatrix calls are lowered.
    for (unsigned I = 0; I < Phi->getNumIncomingValues(); ++I)
      NewPhi->addIncoming(PoisonValue::get(VecTy), Phi->getIncomingBlock(I));
    ValueMap[Phi] = NewPhi;
  }

  // Helper to resolve a value through the ValueMap. If the value was
  // previously a cooperative matrix result, return its concrete replacement.
  auto resolve = [&](Value *V) -> Value * {
    auto It = ValueMap.find(V);
    return It != ValueMap.end() ? It->second : V;
  };

  for (CallInst *CI : CoopCalls) {
    Function *Callee = CI->getCalledFunction();
    auto IID = Callee->getIntrinsicID();
    IRBuilder<> Builder(CI);
    Value *Replacement = nullptr;

    switch (IID) {
    case Intrinsic::coopmatrix_construct:
      Replacement = lowerConstruct(CI, ST, Builder);
      break;
    case Intrinsic::coopmatrix_length:
      Replacement = lowerLength(CI, ST);
      break;
    case Intrinsic::coopmatrix_muladd: {
      // Resolve matrix operands through the ValueMap. The call itself
      // still references the original TargetExtType values, but we pass
      // the concrete vector replacements to the lowering function.
      Value *A = resolve(CI->getArgOperand(0));
      Value *B = resolve(CI->getArgOperand(1));
      Value *C = resolve(CI->getArgOperand(2));
      Replacement = lowerMulAdd(CI, A, B, C, ST, Builder);
      break;
    }
    case Intrinsic::coopmatrix_load:
      Replacement = lowerLoad(CI, ST, Builder);
      break;
    case Intrinsic::coopmatrix_store: {
      Value *Matrix = resolve(CI->getArgOperand(0));
      lowerStore(CI, Matrix, ST, Builder);
      break;
    }
    // --- Set 2: element-wise / metadata ops ---
    case Intrinsic::coopmatrix_unary:
      Replacement = lowerUnary(CI, resolve(CI->getArgOperand(0)), Builder);
      break;
    case Intrinsic::coopmatrix_binary:
      Replacement = lowerBinary(CI, resolve(CI->getArgOperand(0)),
                                resolve(CI->getArgOperand(1)), Builder);
      break;
    case Intrinsic::coopmatrix_convert:
      Replacement = lowerConvert(CI, resolve(CI->getArgOperand(0)), ST, Builder);
      break;
    case Intrinsic::coopmatrix_extract:
      Replacement = lowerExtract(CI, resolve(CI->getArgOperand(0)), Builder);
      break;
    case Intrinsic::coopmatrix_insert:
      Replacement = lowerInsert(CI, resolve(CI->getArgOperand(0)),
                                CI->getArgOperand(1), Builder);
      break;
    case Intrinsic::coopmatrix_get_coord:
      Replacement = lowerGetCoord(CI, ST, Builder);
      break;
    case Intrinsic::coopmatrix_prefetch:
      lowerPrefetch(CI, Builder);
      break;
    // --- Set 3: performance / sparse / checked ops ---
    case Intrinsic::coopmatrix_muladd_ext: {
      Value *A = resolve(CI->getArgOperand(0));
      Value *B = resolve(CI->getArgOperand(1));
      Value *C = resolve(CI->getArgOperand(2));
      Replacement = lowerMulAddExt(CI, A, B, C, ST, Builder);
      break;
    }
    case Intrinsic::coopmatrix_load_checked:
      Replacement = lowerLoadChecked(CI, ST, Builder);
      break;
    case Intrinsic::coopmatrix_store_checked:
      lowerStoreChecked(CI, resolve(CI->getArgOperand(0)), ST, Builder);
      break;
    // Reduce, sparse, and scaled have no direct HW equivalent yet.
    case Intrinsic::coopmatrix_reduce:
      report_fatal_error(
          "coopmatrix.reduce: cross-lane reduction not yet implemented "
          "in AMDGPULowerCooperativeMatrix; requires amdgcn_ds_swizzle/"
          "permlane HW intrinsics (follow-up task)");
    case Intrinsic::coopmatrix_muladd_sparse:
      report_fatal_error(
          "coopmatrix.muladd_sparse: sparse MMA not yet implemented "
          "in AMDGPULowerCooperativeMatrix; requires amdgcn_smfmac "
          "intrinsics (follow-up task)");
    case Intrinsic::coopmatrix_muladd_scaled:
      report_fatal_error(
          "coopmatrix.muladd_scaled: block-scaled MMA not yet implemented "
          "in AMDGPULowerCooperativeMatrix; requires amdgcn_mfma_scale "
          "intrinsics (follow-up task)");
    default:
      llvm_unreachable("unexpected cooperative matrix intrinsic");
    }

    if (Replacement) {
      // Map the old TargetExtType value to the new concrete vector value.
      ValueMap[CI] = Replacement;
      // Do NOT call replaceAllUsesWith here -- remaining uses are other
      // coopmatrix intrinsics that we will resolve via the ValueMap.
      // However, non-coopmatrix uses (like returns or stores) DO need to
      // be replaced. We handle this by replacing all uses after the loop.
    }
    Changed = true;
  }

  // Patch PHI incoming values now that all coopmatrix calls are lowered.
  for (PHINode *Phi : CoopPhis) {
    PHINode *NewPhi = cast<PHINode>(ValueMap[Phi]);
    for (unsigned I = 0; I < Phi->getNumIncomingValues(); ++I) {
      Value *Inc = resolve(Phi->getIncomingValue(I));
      NewPhi->setIncomingValue(I, Inc);
    }
  }

  // Clean up old PHI nodes.
  for (PHINode *Phi : CoopPhis) {
    if (!Phi->use_empty())
      Phi->replaceAllUsesWith(UndefValue::get(Phi->getType()));
    Phi->eraseFromParent();
  }

  // Erase the dead cooperative matrix intrinsic calls in reverse order so
  // downstream consumers (muladd) are erased before their operand producers
  // (construct). For same-type values (like coopmatrix.length → i32), RAUW
  // with the replacement. For type-mismatched values (TargetExtType → vector),
  // remaining uses are from not-yet-erased coopmatrix calls — replace with
  // undef to break the use chain before erasing.
  for (auto It = CoopCalls.rbegin(); It != CoopCalls.rend(); ++It) {
    CallInst *CI = *It;
    auto MapIt = ValueMap.find(CI);
    if (MapIt != ValueMap.end() && CI->getType() == MapIt->second->getType())
      CI->replaceAllUsesWith(MapIt->second);
    else if (!CI->use_empty())
      CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
    CI->eraseFromParent();
  }

  return Changed;
}

bool AMDGPULowerCooperativeMatrix::run(Module &M, const TargetMachine &TM) {
  if (!hasCoopMatrixIntrinsics(M))
    return false;

  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);

    Changed |= processFunction(F, ST);
  }
  return Changed;
}

bool AMDGPULowerCooperativeMatrix::runOnModule(Module &M) {
  TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  const TargetMachine &TM = TPC.getTM<TargetMachine>();
  return run(M, TM);
}

char AMDGPULowerCooperativeMatrix::ID = 0;

char &llvm::AMDGPULowerCooperativeMatrixID =
    AMDGPULowerCooperativeMatrix::ID;

void AMDGPULowerCooperativeMatrix::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
}

#define PASS_DESC "Lower cooperative matrix operations to AMDGPU intrinsics"
INITIALIZE_PASS_BEGIN(AMDGPULowerCooperativeMatrix, DEBUG_TYPE, PASS_DESC,
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPULowerCooperativeMatrix, DEBUG_TYPE, PASS_DESC,
                    false, false)
#undef PASS_DESC

ModulePass *llvm::createAMDGPULowerCooperativeMatrixPass() {
  return new AMDGPULowerCooperativeMatrix();
}

PreservedAnalyses
AMDGPULowerCooperativeMatrixPass::run(Module &M, ModuleAnalysisManager &MA) {
  return AMDGPULowerCooperativeMatrix().run(M, TM)
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}
