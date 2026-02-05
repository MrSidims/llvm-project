; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; Verify that srem with wide vectors (> 16 elements) is legalized correctly.
; The legalizer splits the <32 x i8> srem via fewerElements into
; two <16 x i8> srems. The spv_store intrinsic scalarizes the wide
; result vector for the store.

; CHECK-DAG: %[[#INT8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#VEC32:]] = OpTypeVector %[[#INT8]] 32
; CHECK-DAG: %[[#PTR:]] = OpTypePointer Function %[[#INT8]]

; CHECK-LABEL: OpFunction
; CHECK: OpStore %[[#]] %[[#]] Aligned 32
; CHECK: OpStore %[[#]] %[[#]] Aligned 1
; CHECK: OpReturn
; CHECK: OpFunctionEnd

define spir_kernel void @wide_vector_srem(ptr addrspace(4) %out) {
entry:
  %B = srem <32 x i8> splat (i8 127), splat (i8 1)
  store <32 x i8> %B, ptr null, align 32
  ret void
}
