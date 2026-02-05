; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; Verify that icmp with wide vectors (> 16 elements) is legalized correctly.
; The legalizer splits the <32 x i32> comparison via fewerElements into
; two <16 x i32> comparisons, and the spv_store intrinsic scalarizes
; the wide boolean result vector.

; Wide vector icmp is split into two <16 x i32> comparisons
; CHECK: OpUGreaterThan %[[#]] %[[#]] %[[#]]
; CHECK: OpUGreaterThan %[[#]] %[[#]] %[[#]]
; Scalarized stores of the result
; CHECK: OpCompositeExtract
; CHECK: OpStore
; CHECK: OpReturn

define spir_func void @wide_vector_icmp_ugt() {
entry:
  %C = icmp ugt <32 x i32> splat (i32 42), zeroinitializer
  store <32 x i1> %C, ptr null, align 4
  ret void
}

define spir_func void @wide_vector_icmp_eq() {
entry:
  %C = icmp eq <32 x i32> splat (i32 7), splat (i32 7)
  store <32 x i1> %C, ptr null, align 4
  ret void
}
