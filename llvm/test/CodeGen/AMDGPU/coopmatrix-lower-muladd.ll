; RUN: opt -S -mcpu=gfx1200 -mattr=+wavefrontsize32 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=GFX12-W32
; RUN: opt -S -mcpu=gfx908 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=GFX908

target triple = "amdgcn-amd-amdhsa"

; ============================================================================
; Test 1: coopmatrix.construct (splat) f32 16x16 accumulator
; ============================================================================

define void @test_construct_f32_16x16(float %scalar) {
; GFX12-W32-LABEL: define void @test_construct_f32_16x16(
; GFX12-W32:         insertelement <8 x float> poison, float %scalar, i64 0
; GFX12-W32:         insertelement <8 x float> {{.*}}, float %scalar, i64 7
; GFX12-W32-NEXT:    ret void
;
; GFX908-LABEL: define void @test_construct_f32_16x16(
; GFX908:         insertelement <4 x float> poison, float %scalar, i64 0
; GFX908:         insertelement <4 x float> {{.*}}, float %scalar, i64 3
; GFX908-NEXT:    ret void
;
  %mat = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          float %scalar, i32 3, i32 16, i32 16, i32 2)
  ret void
}

; ============================================================================
; Test 2: coopmatrix.length 16x16 accumulator
; ============================================================================

define i32 @test_length_16x16() {
; GFX12-W32-LABEL: define i32 @test_length_16x16(
; GFX12-W32-NEXT:    ret i32 8
;
; GFX908-LABEL: define i32 @test_length_16x16(
; GFX908-NEXT:    ret i32 4
;
  %len = call i32 @llvm.coopmatrix.length(i32 3, i32 16, i32 16, i32 2)
  ret i32 %len
}

; ============================================================================
; Test 3: construct + muladd f16*f16+f32 16x16x16
; gfx12 w32 -> wmma_f32_16x16x16_f16
; gfx908    -> mfma_f32_16x16x16f16
; ============================================================================

define void @test_muladd_f16_f32_16x16x16(half %sa, half %sb, float %sc) {
; GFX12-W32-LABEL: define void @test_muladd_f16_f32_16x16x16(
; GFX12-W32:         [[WMMA:%.*]] = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(
; GFX12-W32-NEXT:    ret void
;
; GFX908-LABEL: define void @test_muladd_f16_f32_16x16x16(
; GFX908:         [[MFMA:%.*]] = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(
; GFX908-NEXT:    ret void
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
          i32 0, i32 3, i32 16, i32 16, i32 16)
  ret void
}

declare i32 @llvm.coopmatrix.length(i32, i32, i32, i32)
