; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; Verify that cooperative matrix intrinsics survive an IR roundtrip.

; === Declarations with full type (known dimensions) ===

; CHECK: declare target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) @llvm.coopmatrix.load
declare target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0)
    @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i32_3_12_48_0t.p3.i64(
        ptr addrspace(3), i32 immarg, i64,
        i32, i32, i32, i32)

; CHECK: declare void @llvm.coopmatrix.store
declare void
    @llvm.coopmatrix.store.tspirv.CooperativeMatrixKHR_i32_3_12_12_2t.p3.i64(
        target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2),
        ptr addrspace(3), i32 immarg, i64,
        i32, i32, i32, i32)

; CHECK: declare target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @llvm.coopmatrix.muladd
declare target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2)
    @llvm.coopmatrix.muladd.tspirv.CooperativeMatrixKHR_i32_3_12_12_2t.tspirv.CooperativeMatrixKHR_i32_3_12_48_0t.tspirv.CooperativeMatrixKHR_i32_3_48_12_1t(
        target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0),
        target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1),
        target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2),
        i32 immarg,
        i32, i32, i32, i32)

; CHECK: declare i32 @llvm.coopmatrix.length(i32, i32, i32, i32)
declare i32 @llvm.coopmatrix.length(i32, i32, i32, i32)

; CHECK: declare target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @llvm.coopmatrix.construct
declare target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2)
    @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_i32_3_12_12_2t.i32(
        i32, i32, i32, i32, i32)

; === Slim type (spec-constant dimensions) ===

; CHECK: declare target("spirv.CooperativeMatrixKHR", i32) @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i32t.p3.i64
declare target("spirv.CooperativeMatrixKHR", i32)
    @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i32t.p3.i64(
        ptr addrspace(3), i32 immarg, i64,
        i32, i32, i32, i32)

define void @test_coopmatrix_known_dims(ptr addrspace(3) %ptr, i64 %stride, i64 %N) {
entry:
  ; CHECK: %mat_c = call target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @llvm.coopmatrix.construct
  %mat_c = call target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_i32_3_12_12_2t.i32(
          i32 0, i32 3, i32 12, i32 12, i32 2)

  ; CHECK: %mat_a = call target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) @llvm.coopmatrix.load
  %mat_a = call target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i32_3_12_48_0t.p3.i64(
          ptr addrspace(3) %ptr, i32 0, i64 %stride,
          i32 3, i32 12, i32 48, i32 0)

  ; CHECK: %len = call i32 @llvm.coopmatrix.length(i32 3, i32 12, i32 48, i32 0)
  %len = call i32 @llvm.coopmatrix.length(i32 3, i32 12, i32 48, i32 0)

  ; CHECK: call void @llvm.coopmatrix.store
  call void
      @llvm.coopmatrix.store.tspirv.CooperativeMatrixKHR_i32_3_12_12_2t.p3.i64(
          target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %mat_c,
          ptr addrspace(3) %ptr, i32 0, i64 %N,
          i32 3, i32 12, i32 12, i32 2)

  ret void
}

define void @test_coopmatrix_slim_type(ptr addrspace(3) %ptr, i64 %stride,
                                       i32 %scope, i32 %rows, i32 %cols) {
entry:
  ; CHECK: %mat = call target("spirv.CooperativeMatrixKHR", i32) @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i32t.p3.i64
  %mat = call target("spirv.CooperativeMatrixKHR", i32)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i32t.p3.i64(
          ptr addrspace(3) %ptr, i32 0, i64 %stride,
          i32 %scope, i32 %rows, i32 %cols, i32 0)

  ; CHECK: %len = call i32 @llvm.coopmatrix.length(i32 %scope, i32 %rows, i32 %cols, i32 0)
  %len = call i32 @llvm.coopmatrix.length(i32 %scope, i32 %rows, i32 %cols, i32 0)

  ret void
}
