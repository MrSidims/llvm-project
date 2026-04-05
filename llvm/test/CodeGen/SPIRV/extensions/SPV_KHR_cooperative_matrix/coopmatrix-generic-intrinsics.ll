; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix %s -o - | FileCheck %s
; Verify that the generic llvm.coopmatrix.* intrinsics lower to SPIR-V
; cooperative matrix instructions (full type, compile-time-known dimensions).

; CHECK: OpCapability CooperativeMatrixKHR
; CHECK: OpExtension "SPV_KHR_cooperative_matrix"

; CHECK: OpTypeCooperativeMatrixKHR
; CHECK: OpCooperativeMatrixLoadKHR
; CHECK: OpCooperativeMatrixLoadKHR
; CHECK: OpCompositeConstruct
; CHECK: OpCooperativeMatrixMulAddKHR
; CHECK: OpCooperativeMatrixStoreKHR

define spir_kernel void @test_load_muladd_store(ptr addrspace(3) %ptr, i64 %stride) {
entry:
  %a = call target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i32_3_12_48_0t.p3.i64(
          ptr addrspace(3) %ptr, i32 0, i64 %stride,
          i32 3, i32 12, i32 48, i32 0)

  %b = call target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i32_3_48_12_1t.p3.i64(
          ptr addrspace(3) %ptr, i32 0, i64 %stride,
          i32 3, i32 48, i32 12, i32 1)

  %c = call target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_i32_3_12_12_2t.i32(
          i32 0, i32 3, i32 12, i32 12, i32 2)

  %d = call target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2)
      @llvm.coopmatrix.muladd.tspirv.CooperativeMatrixKHR_i32_3_12_12_2t.tspirv.CooperativeMatrixKHR_i32_3_12_48_0t.tspirv.CooperativeMatrixKHR_i32_3_48_12_1t(
          target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) %a,
          target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1) %b,
          target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %c,
          i32 0, i32 3, i32 12, i32 12, i32 48)

  call void
      @llvm.coopmatrix.store.tspirv.CooperativeMatrixKHR_i32_3_12_12_2t.p3.i64(
          target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %d,
          ptr addrspace(3) %ptr, i32 0, i64 %stride,
          i32 3, i32 12, i32 12, i32 2)

  ret void
}

; CHECK: OpCooperativeMatrixLengthKHR
define spir_func i32 @test_length() {
entry:
  %len = call i32 @llvm.coopmatrix.length(i32 3, i32 12, i32 48, i32 0)
  ret i32 %len
}

declare i32 @llvm.coopmatrix.length(i32, i32, i32, i32)
