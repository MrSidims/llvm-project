; RUN: opt -S -mcpu=gfx1100 -mattr=+wavefrontsize32 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=GFX11-W32

target triple = "amdgcn-amd-amdhsa"

; ============================================================================
; Test 1: WMMA v1 f16*f16+f32 16x16x16 muladd
; ============================================================================

define void @test_wmma_v1_f16_muladd(half %sa, half %sb, float %sc) {
; GFX11-W32-LABEL: define void @test_wmma_v1_f16_muladd(
; GFX11-W32:         [[WMMA:%.*]] = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(
; GFX11-W32-NEXT:    ret void
;
  %a = call target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f16_3_16_16_0t.f16(
          half %sa, i32 3, i32 16, i32 16, i32 0)
  %b = call target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 1)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f16_3_16_16_1t.f16(
          half %sb, i32 3, i32 16, i32 16, i32 1)
  %c = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          float %sc, i32 3, i32 16, i32 16, i32 2)
  %result = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.muladd.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.tspirv.CooperativeMatrixKHR_f16_3_16_16_0t.tspirv.CooperativeMatrixKHR_f16_3_16_16_1t(
          target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0) %a,
          target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 1) %b,
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %c,
          i32 0, i32 3, i32 16, i32 16, i32 16, i32 0, i32 0, i32 0)
  ret void
}

; ============================================================================
; Test 2: WMMA v1 i8*i8+i32 16x16x16 muladd
; ============================================================================

define void @test_wmma_v1_i8_muladd(i8 %sa, i8 %sb, i32 %sc) {
; GFX11-W32-LABEL: define void @test_wmma_v1_i8_muladd(
; GFX11-W32:         call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v2i32(
; GFX11-W32-NEXT:    ret void
;
  %a = call target("spirv.CooperativeMatrixKHR", i8, 3, 16, 16, 0)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_i8_3_16_16_0t.i8(
          i8 %sa, i32 3, i32 16, i32 16, i32 0)
  %b = call target("spirv.CooperativeMatrixKHR", i8, 3, 16, 16, 1)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_i8_3_16_16_1t.i8(
          i8 %sb, i32 3, i32 16, i32 16, i32 1)
  %c = call target("spirv.CooperativeMatrixKHR", i32, 3, 16, 16, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_i32_3_16_16_2t.i32(
          i32 %sc, i32 3, i32 16, i32 16, i32 2)
  %result = call target("spirv.CooperativeMatrixKHR", i32, 3, 16, 16, 2)
      @llvm.coopmatrix.muladd.tspirv.CooperativeMatrixKHR_i32_3_16_16_2t.tspirv.CooperativeMatrixKHR_i8_3_16_16_0t.tspirv.CooperativeMatrixKHR_i8_3_16_16_1t(
          target("spirv.CooperativeMatrixKHR", i8, 3, 16, 16, 0) %a,
          target("spirv.CooperativeMatrixKHR", i8, 3, 16, 16, 1) %b,
          target("spirv.CooperativeMatrixKHR", i32, 3, 16, 16, 2) %c,
          i32 0, i32 3, i32 16, i32 16, i32 16, i32 0, i32 0, i32 0)
  ret void
}

; ============================================================================
; Test 3: WMMA v1 load uses stride-2 layout (elements at rows 0,2,4,...,14)
; ============================================================================

define void @test_wmma_v1_load(ptr %ptr) {
; GFX11-W32-LABEL: define void @test_wmma_v1_load(
; GFX11-W32:         call i32 @llvm.amdgcn.mbcnt.lo
; Verify vector load is emitted (not per-element scalar loads):
; GFX11-W32:         load <8 x float>
; GFX11-W32:         ret void
;
  %mat = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.p0(
          ptr %ptr, i32 0, i32 16, i32 3, i32 16, i32 16, i32 2)
  ret void
}

; ============================================================================
; Test 4: WMMA v1 length returns 8
; ============================================================================

define i32 @test_wmma_v1_length() {
; GFX11-W32-LABEL: define i32 @test_wmma_v1_length(
; GFX11-W32-NEXT:    ret i32 8
;
  %len = call i32 @llvm.coopmatrix.length(i32 3, i32 16, i32 16, i32 2)
  ret i32 %len
}

declare i32 @llvm.coopmatrix.length(i32, i32, i32, i32)
