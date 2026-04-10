; RUN: opt -S -mcpu=gfx1200 -mattr=+wavefrontsize32 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=GFX12-W32
; RUN: opt -S -mcpu=gfx908 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=GFX908

target triple = "amdgcn-amd-amdhsa"

; ============================================================================
; Test 1: Vectorized load of f32 16x16 accumulator (WMMA v2)
; Should emit a single <8 x float> load, not 8 scalar loads.
; ============================================================================

define void @test_vector_load_f32_wmma(ptr %ptr) {
; GFX12-W32-LABEL: define void @test_vector_load_f32_wmma(
; GFX12-W32:         load <8 x float>
; GFX12-W32-NOT:     load float
; GFX12-W32:         ret void
;
  %mat = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.p0(
          ptr %ptr, i32 0, i32 16, i32 3, i32 16, i32 16, i32 2)
  ret void
}

; ============================================================================
; Test 2: Vectorized store of f32 16x16 accumulator (WMMA v2)
; Should emit a single <8 x float> store, not 8 scalar stores.
; ============================================================================

define void @test_vector_store_f32_wmma(ptr %ptr, float %sc) {
; GFX12-W32-LABEL: define void @test_vector_store_f32_wmma(
; GFX12-W32:         store <8 x float>
; GFX12-W32-NOT:     store float
; GFX12-W32:         ret void
;
  %mat = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          float %sc, i32 3, i32 16, i32 16, i32 2)
  call void @llvm.coopmatrix.store.p0.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %mat,
          ptr %ptr, i32 0, i32 16, i32 3, i32 16, i32 16, i32 2)
  ret void
}

; ============================================================================
; Test 3: Vectorized load of <4 x float> MFMA 16x16 accumulator
; ============================================================================

define void @test_vector_load_f32_mfma_16x16(ptr %ptr) {
; GFX908-LABEL: define void @test_vector_load_f32_mfma_16x16(
; GFX908:         load <4 x float>
; GFX908-NOT:     load float
; GFX908:         ret void
;
  %mat = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.p0(
          ptr %ptr, i32 0, i32 16, i32 3, i32 16, i32 16, i32 2)
  ret void
}

; ============================================================================
; Test 4: Vectorized load of <16 x float> MFMA 32x32 accumulator
; ============================================================================

define void @test_vector_load_f32_mfma_32x32(ptr %ptr) {
; GFX908-LABEL: define void @test_vector_load_f32_mfma_32x32(
; GFX908:         load <16 x float>
; GFX908-NOT:     load float
; GFX908:         ret void
;
  %mat = call target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_f32_3_32_32_2t.p0(
          ptr %ptr, i32 0, i32 32, i32 3, i32 32, i32 32, i32 2)
  ret void
}
