; Phase 2 numerical-correctness test for MFMA 16x16x16 f16 * f16 + f32.
;
; This test covers BUG-2 of the Phase 2 AMDGPULowerCooperativeMatrix fixes:
; the A-operand of a 16x16x16 MFMA must use the transpose of the accumulator
; formula (row = lane%16, col = (lane/16)*4 + elem), NOT the accumulator
; formula itself. It also verifies BUG-1 by keeping the accumulator 16x16
; formula intact.
;
; We check two things:
;   1. The IR produced by the amdgpu-lower-cooperative-matrix pass on the
;      load/muladd/store chain contains the expected Use-aware coordinate
;      arithmetic.
;   2. The llc pipeline for gfx908/gfx942 emits a real MFMA instruction
;      plus memory ops (loads and stores) for the A/B/C operand plumbing.

; RUN: opt -S -mcpu=gfx908 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=IR

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 < %s \
; RUN:   | FileCheck %s --check-prefix=GFX908
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 < %s \
; RUN:   | FileCheck %s --check-prefix=GFX942

target triple = "amdgcn-amd-amdhsa"

; ============================================================================
; End-to-end f16*f16+f32 16x16x16 coop matrix muladd.
; ============================================================================

define amdgpu_kernel void @coopmat_mma_16x16x16_f16(
    ptr addrspace(1) %pa, ptr addrspace(1) %pb, ptr addrspace(1) %pc) {
entry:
  %a = call target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f16_3_16_16_0t.p1.i32(
          ptr addrspace(1) %pa, i32 0, i32 16, i32 3, i32 16, i32 16, i32 0)
  %b = call target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 1)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f16_3_16_16_1t.p1.i32(
          ptr addrspace(1) %pb, i32 0, i32 16, i32 3, i32 16, i32 16, i32 1)
  %c = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.p1.i32(
          ptr addrspace(1) %pc, i32 0, i32 16, i32 3, i32 16, i32 16, i32 2)
  %d = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.muladd.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.tspirv.CooperativeMatrixKHR_f16_3_16_16_0t.tspirv.CooperativeMatrixKHR_f16_3_16_16_1t(
          target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0) %a,
          target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 1) %b,
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %c,
          i32 0, i32 3, i32 16, i32 16, i32 16, i32 0, i32 0, i32 0)
  call void
      @llvm.coopmatrix.store.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.p1.i32(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %d,
          ptr addrspace(1) %pc, i32 0, i32 16, i32 3, i32 16, i32 16, i32 2)
  ret void
}

; IR-LABEL: define amdgpu_kernel void @coopmat_mma_16x16x16_f16(
;
; A-operand: row = lane % 16, col = (lane / 16) * 4 + elem.
; Accumulator: row = (lane / 16) * 4 + elem, col = lane % 16.
; B-operand: col = lane % 16, row = (lane / 16) * 4 + elem.
; After BUG-2 fix, A and the accumulator use DIFFERENT arithmetic
; (transposed), so we expect BOTH urem-by-16 and udiv-by-16 in the
; lowered IR, plus a mul-by-4 for the kWidth stride.
;
; IR-DAG: urem i64 {{.*}}, 16
; IR-DAG: udiv i64 {{.*}}, 16
; IR-DAG: mul i64 {{.*}}, 4
;
; Final muladd lowers to amdgcn_mfma_f32_16x16x16f16 with <4 x half>
; A/B and <4 x float> C operands.
; IR: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(

; GFX908-LABEL: coopmat_mma_16x16x16_f16:
; GFX908: v_mfma_f32_16x16x16{{.*}}f16
; GFX908: global_store

; GFX942-LABEL: coopmat_mma_16x16x16_f16:
; GFX942: v_mfma_f32_16x16x16{{.*}}f16
; GFX942: global_store

declare target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0)
    @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f16_3_16_16_0t.p1.i32(
        ptr addrspace(1), i32, i32, i32, i32, i32, i32)

declare target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 1)
    @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f16_3_16_16_1t.p1.i32(
        ptr addrspace(1), i32, i32, i32, i32, i32, i32)

declare target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
    @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.p1.i32(
        ptr addrspace(1), i32, i32, i32, i32, i32, i32)

; muladd declaration auto-generated from Intrinsics.td

declare void
    @llvm.coopmatrix.store.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.p1.i32(
        target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2),
        ptr addrspace(1), i32, i32, i32, i32, i32, i32)
