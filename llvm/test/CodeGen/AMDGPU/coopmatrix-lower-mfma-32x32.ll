; RUN: opt -S -mcpu=gfx908 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=GFX908

target triple = "amdgcn-amd-amdhsa"

; ============================================================================
; coopmatrix.muladd f16*f16+f32 32x32x8 (MFMA)
; gfx908 -> mfma_f32_32x32x8f16
; ============================================================================

define void @test_muladd_f16_f32_32x32x8(half %sa, half %sb, float %sc) {
; GFX908-LABEL: define void @test_muladd_f16_f32_32x32x8(
; GFX908:         [[MFMA:%.*]] = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(
; GFX908-NEXT:    ret void
;
  %a = call target("spirv.CooperativeMatrixKHR", half, 3, 32, 8, 0)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f16_3_32_8_0t.f16(
          half %sa, i32 3, i32 32, i32 8, i32 0)
  %b = call target("spirv.CooperativeMatrixKHR", half, 3, 8, 32, 1)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f16_3_8_32_1t.f16(
          half %sb, i32 3, i32 8, i32 32, i32 1)
  %c = call target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_32_32_2t.f32(
          float %sc, i32 3, i32 32, i32 32, i32 2)
  %result = call target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2)
      @llvm.coopmatrix.muladd.tspirv.CooperativeMatrixKHR_f32_3_32_32_2t.tspirv.CooperativeMatrixKHR_f16_3_32_8_0t.tspirv.CooperativeMatrixKHR_f16_3_8_32_1t(
          target("spirv.CooperativeMatrixKHR", half, 3, 32, 8, 0) %a,
          target("spirv.CooperativeMatrixKHR", half, 3, 8, 32, 1) %b,
          target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2) %c,
          i32 0, i32 3, i32 32, i32 32, i32 8)
  ret void
}

; ============================================================================
; coopmatrix.length 32x32 accumulator
; ============================================================================

define i32 @test_length_32x32() {
; GFX908-LABEL: define i32 @test_length_32x32(
; GFX908-NEXT:    ret i32 16
;
  %len = call i32 @llvm.coopmatrix.length(i32 3, i32 32, i32 32, i32 2)
  ret i32 %len
}

declare i32 @llvm.coopmatrix.length(i32, i32, i32, i32)
