; RUN: opt -S -mcpu=gfx942 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=GFX942
; RUN: opt -S -mcpu=gfx950 \
; RUN:   -passes=amdgpu-lower-cooperative-matrix < %s \
; RUN:   | FileCheck %s --check-prefix=GFX950

target triple = "amdgcn-amd-amdhsa"

; Test FP8 MFMA cooperative matrix lowering.
; gfx942 and gfx950 both use amdgcn_mfma_f32_16x16x32_fp8_fp8
; (gfx942 interprets as FNUZ, gfx950 as OCP — same intrinsic name).

; GFX942-LABEL: define void @test_fp8_16x16x32
; GFX942: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(
; GFX950-LABEL: define void @test_fp8_16x16x32
; GFX950: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(
define void @test_fp8_16x16x32(ptr addrspace(1) %ptrA, ptr addrspace(1) %ptrB, ptr addrspace(1) %ptrC) {
entry:
  %c = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          float 0.0, i32 3, i32 16, i32 16, i32 2)
  %a = call target("spirv.CooperativeMatrixKHR", i8, 3, 16, 32, 0)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i8_3_16_32_0t.p1.i64(
          ptr addrspace(1) %ptrA, i32 0, i64 32, i32 3, i32 16, i32 32, i32 0)
  %b = call target("spirv.CooperativeMatrixKHR", i8, 3, 32, 16, 1)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i8_3_32_16_1t.p1.i64(
          ptr addrspace(1) %ptrB, i32 0, i64 16, i32 3, i32 32, i32 16, i32 1)
  %d = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.muladd.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.tspirv.CooperativeMatrixKHR_i8_3_16_32_0t.tspirv.CooperativeMatrixKHR_i8_3_32_16_1t(
          target("spirv.CooperativeMatrixKHR", i8, 3, 16, 32, 0) %a,
          target("spirv.CooperativeMatrixKHR", i8, 3, 32, 16, 1) %b,
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %c,
          i32 0, i32 3, i32 16, i32 16, i32 32, i32 0, i32 0, i32 0)
  ret void
}

; GFX942-LABEL: define void @test_fp8_32x32x16
; GFX942: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8(
; GFX950-LABEL: define void @test_fp8_32x32x16
; GFX950: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8(
define void @test_fp8_32x32x16(ptr addrspace(1) %ptrA, ptr addrspace(1) %ptrB, ptr addrspace(1) %ptrC) {
entry:
  %c = call target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_f32_3_32_32_2t.f32(
          float 0.0, i32 3, i32 32, i32 32, i32 2)
  %a = call target("spirv.CooperativeMatrixKHR", i8, 3, 32, 16, 0)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i8_3_32_16_0t.p1.i64(
          ptr addrspace(1) %ptrA, i32 0, i64 16, i32 3, i32 32, i32 16, i32 0)
  %b = call target("spirv.CooperativeMatrixKHR", i8, 3, 16, 32, 1)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i8_3_16_32_1t.p1.i64(
          ptr addrspace(1) %ptrB, i32 0, i64 32, i32 3, i32 16, i32 32, i32 1)
  %d = call target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2)
      @llvm.coopmatrix.muladd.tspirv.CooperativeMatrixKHR_f32_3_32_32_2t.tspirv.CooperativeMatrixKHR_i8_3_32_16_0t.tspirv.CooperativeMatrixKHR_i8_3_16_32_1t(
          target("spirv.CooperativeMatrixKHR", i8, 3, 32, 16, 0) %a,
          target("spirv.CooperativeMatrixKHR", i8, 3, 16, 32, 1) %b,
          target("spirv.CooperativeMatrixKHR", float, 3, 32, 32, 2) %c,
          i32 0, i32 3, i32 32, i32 32, i32 16, i32 0, i32 0, i32 0)
  ret void
}
