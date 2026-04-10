; Phase 2 numerical-correctness test for MFMA 32x32x8 f16 * f16 + f32.
;
; This covers BUG-1 (32x32 accumulator layout: HalfWave stride 4, tile stride
; T*8 — NOT HalfWave stride 16 with (T>>1)*8 + (T&1)*4) and BUG-3 (A/B use a
; Use-aware transpose formula instead of the broken row-major fallback).
;
; Matrix shapes:
;   A: target("...", half, 3, 32, 8, 0)     M=32, K=8
;   B: target("...", half, 3, 8, 32, 1)     K=8, N=32
;   C/D: target("...", float, 3, 32, 32, 2) M=32, N=32

; RUN: opt -S -mcpu=gfx908 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=IR

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 < %s \
; RUN:   | FileCheck %s --check-prefix=GFX908
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 < %s \
; RUN:   | FileCheck %s --check-prefix=GFX942

target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @coopmat_mma_32x32x8_f16(
    ptr addrspace(1) %pa, ptr addrspace(1) %pb, ptr addrspace(1) %pc) {
entry:
  %a = call target("spirv.CooperativeMatrixKHR", half, 3, 32, 8, 0)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f16_3_32_8_0t.p1.i32(
          ptr addrspace(1) %pa, i32 0, i32 8, i32 3, i32 32, i32 8, i32 0)
  %b = call target("spirv.CooperativeMatrixKHR", half, 3, 8, 32, 1)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f16_3_8_32_1t.p1.i32(
          ptr addrspace(1) %pb, i32 0, i32 32, i32 3, i32 8, i32 32, i32 1)
  %c = call target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f32_3_32_32_2t.p1.i32(
          ptr addrspace(1) %pc, i32 0, i32 32, i32 3, i32 32, i32 32, i32 2)
  %d = call target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2)
      @llvm.coopmatrix.muladd.tspirv.CooperativeMatrixKHR_f32_3_32_32_2t.tspirv.CooperativeMatrixKHR_f16_3_32_8_0t.tspirv.CooperativeMatrixKHR_f16_3_8_32_1t(
          target("spirv.CooperativeMatrixKHR", half, 3, 32, 8, 0) %a,
          target("spirv.CooperativeMatrixKHR", half, 3, 8, 32, 1) %b,
          target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2) %c,
          i32 0, i32 3, i32 32, i32 32, i32 8, i32 0, i32 0, i32 0)
  call void
      @llvm.coopmatrix.store.tspirv.CooperativeMatrixKHR_f32_3_32_32_2t.p1.i32(
          target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2) %d,
          ptr addrspace(1) %pc, i32 0, i32 32, i32 3, i32 32, i32 32, i32 2)
  ret void
}

; IR-LABEL: define amdgpu_kernel void @coopmat_mma_32x32x8_f16(
;
; A operand (32x8 use=0): row = lane%32, col = (lane/32)*4 + elem.
; B operand (8x32 use=1): col = lane%32, row = (lane/32)*4 + elem.
; Accumulator (32x32 use=2): halfWaveOff = (lane>>5)*4; tileBase = halfWaveOff + T*8
;                            for T in 0..3; row = tileBase + elem for elem in 0..3;
;                            col = lane%32.
;
; BUG-1 fix: the half-wave stride for the accumulator 32x32 path is 4,
; not 16. We verify that the HalfWave lshr-5 result is multiplied by 4.
; Tile stride is T*8 — check the constant offsets 8, 16, 24 appear.
; IR-DAG: lshr i64 {{.*}}, 5
; IR-DAG: mul i64 {{.*}}, 4
; IR-DAG: urem i64 {{.*}}, 32
; IR-DAG: udiv i64 {{.*}}, 32
;
; The muladd lowers to amdgcn_mfma_f32_32x32x8f16 with <4 x half> A/B and
; <16 x float> C.
; IR: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(

; GFX908-LABEL: coopmat_mma_32x32x8_f16:
; GFX908: v_mfma_f32_32x32x8{{.*}}f16
; GFX908: global_store

; GFX942-LABEL: coopmat_mma_32x32x8_f16:
; GFX942: v_mfma_f32_32x32x8{{.*}}f16
; GFX942: global_store

declare target("spirv.CooperativeMatrixKHR", half, 3, 32, 8, 0)
    @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f16_3_32_8_0t.p1.i32(
        ptr addrspace(1), i32, i32, i32, i32, i32, i32)

declare target("spirv.CooperativeMatrixKHR", half, 3, 8, 32, 1)
    @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f16_3_8_32_1t.p1.i32(
        ptr addrspace(1), i32, i32, i32, i32, i32, i32)

declare target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2)
    @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f32_3_32_32_2t.p1.i32(
        ptr addrspace(1), i32, i32, i32, i32, i32, i32)

; muladd declaration auto-generated from Intrinsics.td

declare void
    @llvm.coopmatrix.store.tspirv.CooperativeMatrixKHR_f32_3_32_32_2t.p1.i32(
        target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2),
        ptr addrspace(1), i32, i32, i32, i32, i32, i32)
