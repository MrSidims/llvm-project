; RUN: opt -S -mcpu=gfx1200 -mattr=+wavefrontsize32 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=GFX12-W32
; RUN: opt -S -mcpu=gfx908 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=GFX908

target triple = "amdgcn-amd-amdhsa"

; ============================================================================
; Test 1: coopmatrix.unary Negate (op=0) on f32 accumulator
; ============================================================================

define void @test_unary_negate(float %sc) {
; GFX12-W32-LABEL: define void @test_unary_negate(
; GFX12-W32:         fneg <8 x float>
; GFX12-W32:         ret void
;
; GFX908-LABEL: define void @test_unary_negate(
; GFX908:         fneg <4 x float>
; GFX908:         ret void
;
  %mat = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          float %sc, i32 3, i32 16, i32 16, i32 2)
  %neg = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.unary.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %mat,
          i32 0, i32 3, i32 16, i32 16, i32 2)
  ret void
}

; ============================================================================
; Test 2: coopmatrix.unary Abs (op=1) on f32 accumulator
; ============================================================================

define void @test_unary_abs(float %sc) {
; GFX12-W32-LABEL: define void @test_unary_abs(
; GFX12-W32:         call <8 x float> @llvm.fabs.v8f32(
; GFX12-W32:         ret void
;
; GFX908-LABEL: define void @test_unary_abs(
; GFX908:         call <4 x float> @llvm.fabs.v4f32(
; GFX908:         ret void
;
  %mat = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          float %sc, i32 3, i32 16, i32 16, i32 2)
  %abs = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.unary.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %mat,
          i32 1, i32 3, i32 16, i32 16, i32 2)
  ret void
}

; ============================================================================
; Test 3: coopmatrix.binary Add (op=0) on f32 accumulator
; ============================================================================

define void @test_binary_add(float %sa, float %sb) {
; GFX12-W32-LABEL: define void @test_binary_add(
; GFX12-W32:         fadd <8 x float>
; GFX12-W32:         ret void
;
; GFX908-LABEL: define void @test_binary_add(
; GFX908:         fadd <4 x float>
; GFX908:         ret void
;
  %a = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          float %sa, i32 3, i32 16, i32 16, i32 2)
  %b = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          float %sb, i32 3, i32 16, i32 16, i32 2)
  %sum = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.binary.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %a,
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %b,
          i32 0, i32 3, i32 16, i32 16, i32 2)
  ret void
}

; ============================================================================
; Test 4: coopmatrix.extract (constant index)
; ============================================================================

define float @test_extract_const(float %sc) {
; GFX12-W32-LABEL: define float @test_extract_const(
; GFX12-W32:         extractelement <8 x float> {{.*}}, i32 2
; GFX12-W32:         ret float
;
; GFX908-LABEL: define float @test_extract_const(
; GFX908:         extractelement <4 x float> {{.*}}, i32 2
; GFX908:         ret float
;
  %mat = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          float %sc, i32 3, i32 16, i32 16, i32 2)
  %elem = call float
      @llvm.coopmatrix.extract.f32.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %mat,
          i32 2, i32 3, i32 16, i32 16, i32 2)
  ret float %elem
}

; ============================================================================
; Test 5: coopmatrix.insert
; ============================================================================

define void @test_insert(float %sc, float %val) {
; GFX12-W32-LABEL: define void @test_insert(
; GFX12-W32:         insertelement <8 x float> {{.*}}, float %val, i32 1
; GFX12-W32:         ret void
;
; GFX908-LABEL: define void @test_insert(
; GFX908:         insertelement <4 x float> {{.*}}, float %val, i32 1
; GFX908:         ret void
;
  %mat = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          float %sc, i32 3, i32 16, i32 16, i32 2)
  %result = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.insert.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %mat,
          float %val, i32 1, i32 3, i32 16, i32 16, i32 2)
  ret void
}

; ============================================================================
; Test 6: coopmatrix.get_coord with constant index
; ============================================================================

define <2 x i32> @test_get_coord() {
; GFX12-W32-LABEL: define <2 x i32> @test_get_coord(
; GFX12-W32:         call i32 @llvm.amdgcn.mbcnt.lo
; GFX12-W32:         ret <2 x i32>
;
; GFX908-LABEL: define <2 x i32> @test_get_coord(
; GFX908:         call i32 @llvm.amdgcn.mbcnt.lo
; GFX908:         ret <2 x i32>
;
  %mat = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          float 1.0, i32 3, i32 16, i32 16, i32 2)
  %coord = call <2 x i32>
      @llvm.coopmatrix.get.coord.v2i32.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %mat,
          i32 0, i32 3, i32 16, i32 16, i32 2)
  ret <2 x i32> %coord
}
