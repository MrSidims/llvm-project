; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that without the extension, we use typed pointers (OpTypePointer).
; This is a regression test to ensure the extension doesn't break normal operation.

; CHECK-NOT: OpCapability UntypedPointersKHR
; CHECK-NOT: OpExtension "SPV_KHR_untyped_pointers"
; CHECK-NOT: OpTypeUntypedPointerKHR
; CHECK-NOT: OpUntypedVariableKHR
; CHECK-NOT: OpUntypedAccessChainKHR

; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: OpTypePointer CrossWorkgroup [[I32]]
; CHECK-DAG: OpTypePointer Function [[I32]]

; Test pointer parameters - should use OpTypePointer
define spir_kernel void @test_kernel(ptr addrspace(1) %input, ptr addrspace(1) %output) {
entry:
  %val = load i32, ptr addrspace(1) %input, align 4
  store i32 %val, ptr addrspace(1) %output, align 4
  ret void
}

; Test alloca - should use OpVariable (not OpUntypedVariableKHR)
; CHECK: OpVariable
define spir_kernel void @test_alloca() {
entry:
  %local = alloca i32, align 4
  store i32 42, ptr %local, align 4
  ret void
}

; Test GEP - should use OpPtrAccessChain (not OpUntypedPtrAccessChainKHR)
; CHECK: OpPtrAccessChain
define spir_kernel void @test_gep(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr i32, ptr addrspace(1) %base, i64 5
  %val = load i32, ptr addrspace(1) %ptr, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}
