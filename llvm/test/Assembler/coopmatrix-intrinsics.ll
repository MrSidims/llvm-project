; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; Verify that cooperative matrix intrinsics survive an IR roundtrip.

; Type aliases used in tests:
;   %mat_a  = target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0)  ; MatrixA
;   %mat_b  = target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1)  ; MatrixB
;   %mat_c  = target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2)  ; Accumulator
;   %mat_f  = target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) ; Float accum
;   %slim   = target("spirv.CooperativeMatrixKHR", i32)                 ; Spec-constant dims

; ============================================================
; Set 1: Baseline operations
; ============================================================

define void @test_baseline(ptr addrspace(3) %ptr, i64 %stride, i64 %N) {
entry:
  ; CHECK: @llvm.coopmatrix.construct
  %mat_c = call target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2)
      @llvm.coopmatrix.construct.tspirv.CooperativeMatrixKHR_i32_3_12_12_2t.i32(
          i32 0, i32 3, i32 12, i32 12, i32 2)

  ; CHECK: @llvm.coopmatrix.load
  %mat_a = call target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i32_3_12_48_0t.p3.i64(
          ptr addrspace(3) %ptr, i32 0, i64 %stride,
          i32 3, i32 12, i32 48, i32 0)

  ; CHECK: @llvm.coopmatrix.length
  %len = call i32 @llvm.coopmatrix.length(i32 3, i32 12, i32 48, i32 0)

  ; CHECK: @llvm.coopmatrix.store
  call void
      @llvm.coopmatrix.store.tspirv.CooperativeMatrixKHR_i32_3_12_12_2t.p3.i64(
          target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %mat_c,
          ptr addrspace(3) %ptr, i32 0, i64 %N,
          i32 3, i32 12, i32 12, i32 2)

  ret void
}

; ============================================================
; Set 1: Slim type (spec-constant dimensions)
; ============================================================

define void @test_slim_type(ptr addrspace(3) %ptr, i64 %stride,
                            i32 %scope, i32 %rows, i32 %cols) {
entry:
  ; CHECK: @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i32t.p3.i64
  %mat = call target("spirv.CooperativeMatrixKHR", i32)
      @llvm.coopmatrix.load.tspirv.CooperativeMatrixKHR_i32t.p3.i64(
          ptr addrspace(3) %ptr, i32 0, i64 %stride,
          i32 %scope, i32 %rows, i32 %cols, i32 0)

  ; CHECK: @llvm.coopmatrix.length(i32 %scope, i32 %rows, i32 %cols, i32 0)
  %len = call i32 @llvm.coopmatrix.length(i32 %scope, i32 %rows, i32 %cols, i32 0)

  ret void
}

; ============================================================
; Set 2: Extended operations
; ============================================================

define void @test_extended(
    target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %matA,
    target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %matB,
    target("spirv.CooperativeMatrixKHR", i32, 3, 16, 16, 2) %imat) {
entry:
  ; CHECK: @llvm.coopmatrix.unary
  %neg = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.unary.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %matA,
          i32 0,  ; Negate
          i32 3, i32 16, i32 16, i32 2)

  ; CHECK: @llvm.coopmatrix.binary
  %sum = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.binary.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %matA,
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %matB,
          i32 0,  ; Add
          i32 3, i32 16, i32 16, i32 2)

  ; CHECK: @llvm.coopmatrix.convert
  %conv = call target("spirv.CooperativeMatrixKHR", i32, 3, 16, 16, 2)
      @llvm.coopmatrix.convert.tspirv.CooperativeMatrixKHR_i32_3_16_16_2t.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %matA,
          i32 3, i32 16, i32 16, i32 2, i32 2)

  ; CHECK: @llvm.coopmatrix.reduce
  %red = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.reduce.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %matA,
          i32 0, i32 0,  ; op=Add, dim=ReduceRow
          i32 3, i32 16, i32 16, i32 2)

  ; CHECK: @llvm.coopmatrix.extract
  %elem = call float
      @llvm.coopmatrix.extract.f32.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %matA,
          i32 0,
          i32 3, i32 16, i32 16, i32 2)

  ; CHECK: @llvm.coopmatrix.insert
  %ins = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.insert.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.f32(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %matA,
          float 1.0, i32 0,
          i32 3, i32 16, i32 16, i32 2)

  ; CHECK: @llvm.coopmatrix.get.coord
  %coord = call <2 x i32>
      @llvm.coopmatrix.get.coord.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %matA,
          i32 0,
          i32 3, i32 16, i32 16, i32 2)

  ret void
}

define void @test_prefetch(ptr addrspace(1) %ptr, i64 %stride) {
entry:
  ; CHECK: @llvm.coopmatrix.prefetch
  call void @llvm.coopmatrix.prefetch.p1.i64(
      ptr addrspace(1) %ptr, i32 16, i32 16,
      i32 0, i32 0, i64 %stride, i32 3)
  ret void
}

; ============================================================
; Set 3: Performance-oriented MMA
; ============================================================

define void @test_muladd_ext(
    target("spirv.CooperativeMatrixKHR", half, 3, 16, 32, 0) %A,
    target("spirv.CooperativeMatrixKHR", half, 3, 32, 16, 1) %B,
    target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %C) {
entry:
  ; CHECK: @llvm.coopmatrix.muladd.ext
  %r = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.muladd.ext.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.tspirv.CooperativeMatrixKHR_f16_3_16_32_0t.tspirv.CooperativeMatrixKHR_f16_3_32_16_1t(
          target("spirv.CooperativeMatrixKHR", half, 3, 16, 32, 0) %A,
          target("spirv.CooperativeMatrixKHR", half, 3, 32, 16, 1) %B,
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %C,
          i32 0, i32 3,  ; operands=0, modifiers=neg_A|neg_B
          i32 3, i32 16, i32 16, i32 32)
  ret void
}

define void @test_muladd_sparse(
    target("spirv.CooperativeMatrixKHR", half, 3, 16, 32, 0) %A,
    target("spirv.CooperativeMatrixKHR", half, 3, 32, 16, 1) %B,
    target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %C,
    i16 %sparsity) {
entry:
  ; CHECK: @llvm.coopmatrix.muladd.sparse
  %r = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.muladd.sparse.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.tspirv.CooperativeMatrixKHR_f16_3_16_32_0t.tspirv.CooperativeMatrixKHR_f16_3_32_16_1t.i16(
          target("spirv.CooperativeMatrixKHR", half, 3, 16, 32, 0) %A,
          target("spirv.CooperativeMatrixKHR", half, 3, 32, 16, 1) %B,
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %C,
          i16 %sparsity,
          i32 0, i32 0,
          i32 3, i32 16, i32 16, i32 32)
  ret void
}

define void @test_muladd_scaled(
    target("spirv.CooperativeMatrixKHR", i8, 3, 16, 128, 0) %A,
    target("spirv.CooperativeMatrixKHR", i8, 3, 128, 16, 1) %B,
    target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %C,
    i32 %scaleA, i32 %scaleB) {
entry:
  ; CHECK: @llvm.coopmatrix.muladd.scaled
  %r = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.muladd.scaled.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.tspirv.CooperativeMatrixKHR_i8_3_16_128_0t.tspirv.CooperativeMatrixKHR_i8_3_128_16_1t.i32.i32(
          target("spirv.CooperativeMatrixKHR", i8, 3, 16, 128, 0) %A,
          target("spirv.CooperativeMatrixKHR", i8, 3, 128, 16, 1) %B,
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %C,
          i32 1, i32 1,  ; a_fmt=FP8_E4M3, b_fmt=FP8_E4M3
          i32 %scaleA, i32 0,  ; a_scale, a_scale_fmt
          i32 %scaleB, i32 0,  ; b_scale, b_scale_fmt
          i32 0,  ; modifiers
          i32 3, i32 16, i32 16, i32 128)
  ret void
}

define void @test_checked_load_store(ptr addrspace(1) %ptr, i64 %stride) {
entry:
  ; CHECK: @llvm.coopmatrix.load.checked
  %mat = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.coopmatrix.load.checked.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.p1.i64(
          ptr addrspace(1) %ptr, i32 0, i32 0,
          i32 100, i32 100, i32 0, i64 %stride,
          i32 3, i32 16, i32 16, i32 2)

  ; CHECK: @llvm.coopmatrix.store.checked
  call void
      @llvm.coopmatrix.store.checked.tspirv.CooperativeMatrixKHR_f32_3_16_16_2t.p1.i64(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %mat,
          ptr addrspace(1) %ptr, i32 0, i32 0,
          i32 100, i32 100, i32 0, i64 %stride,
          i32 3, i32 16, i32 16, i32 2)

  ret void
}

; ============================================================
; Declarations (for explicit declare CHECK patterns)
; ============================================================

declare i32 @llvm.coopmatrix.length(i32, i32, i32, i32)
declare void @llvm.coopmatrix.prefetch.p1.i64(ptr addrspace(1), i32, i32, i32, i32, i64, i32)
