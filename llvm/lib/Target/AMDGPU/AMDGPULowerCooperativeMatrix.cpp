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
  // WMMA (wave32, gfx12+): 8 elements per lane for all matrix operands.
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
    } else {
      // A/B operands for MFMA.
      if (ElemTy->isHalfTy() || ElemTy->isBFloatTy())
        return FixedVectorType::get(ElemTy, 4);
      if (ElemTy->isIntegerTy(8))
        return FixedVectorType::get(ElemTy, 4);
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
  } else {
    report_fatal_error("coopmatrix.load: unsupported subtarget");
  }

  // Load each element: addr = base + row * stride * elem_size + col * elem_size
  Value *Vec = PoisonValue::get(VecTy);
  for (unsigned E = 0; E < VecLen; ++E) {
    auto [Row, Col] = Coords[E];
    Value *ByteOff = Builder.CreateAdd(
        Builder.CreateMul(Row, Builder.CreateMul(Stride64,
                                                  Builder.getInt64(ElemSize))),
        Builder.CreateMul(Col, Builder.getInt64(ElemSize)));
    Value *ElemPtr = Builder.CreateGEP(Builder.getInt8Ty(), Ptr, ByteOff);
    Value *Elem = Builder.CreateAlignedLoad(ElemTy, ElemPtr,
                                            MaybeAlign(ElemSize));
    Vec = Builder.CreateInsertElement(Vec, Elem, E);
  }

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
  } else {
    report_fatal_error("coopmatrix.store: unsupported subtarget");
  }

  // Store each element at its correct position.
  for (unsigned E = 0; E < VecLen; ++E) {
    auto [Row, Col] = Coords[E];
    Value *ByteOff = Builder.CreateAdd(
        Builder.CreateMul(Row, Builder.CreateMul(Stride64,
                                                  Builder.getInt64(ElemSize))),
        Builder.CreateMul(Col, Builder.getInt64(ElemSize)));
    Value *ElemPtr = Builder.CreateGEP(Builder.getInt8Ty(), Ptr, ByteOff);
    Value *Elem = Builder.CreateExtractElement(Matrix, E);
    Builder.CreateAlignedStore(Elem, ElemPtr, MaybeAlign(ElemSize));
  }
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
  // K(7)
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
    // BUG-5 fix: Set 2/3 intrinsics are collected above so they don't
    // silently flow into SelectionDAG, but they are not yet lowered here.
    // Halt compilation with a clear message rather than miscompile.
    // Phase 3d's set2-amdgpu-agent will replace these stubs with real
    // lowerings.
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
      report_fatal_error(Twine("Intrinsic ") +
                         Intrinsic::getBaseName(IID) +
                         " not yet lowered in "
                         "AMDGPULowerCooperativeMatrix; implementation "
                         "coming in Phase 3d");
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
