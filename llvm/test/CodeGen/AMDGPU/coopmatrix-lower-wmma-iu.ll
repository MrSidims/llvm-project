; RUN: opt -S -mcpu=gfx1200 -mattr=+wavefrontsize32 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=GFX12-W32

target triple = "amdgcn-amd-amdhsa"

; ============================================================================
; coopmatrix.muladd i8*i8+i32 16x16x16 (WMMA IU)
; gfx12 w32 -> wmma_i32_16x16x16_iu8
; ============================================================================

define void @test_muladd_i8_i32_16x16x16(i8 %sa, i8 %sb, i32 %sc) {
; GFX12-W32-LABEL: define void @test_muladd_i8_i32_16x16x16(
; GFX12-W32:         [[A_BC:%.*]] = bitcast <8 x i8> {{.*}} to <2 x i32>
; GFX12-W32:         [[B_BC:%.*]] = bitcast <8 x i8> {{.*}} to <2 x i32>
; GFX12-W32:         [[WMMA:%.*]] = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v2i32(i1 false, <2 x i32> [[A_BC]], i1 false, <2 x i32> [[B_BC]], <8 x i32> {{.*}}, i1 false)
; GFX12-W32-NEXT:    ret void
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
          i32 0, i32 3, i32 16, i32 16, i32 16)
  ret void
}
