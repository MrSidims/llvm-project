; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[CROSS_PTR:%[0-9]+]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-DAG: [[FUNC_PTR:%[0-9]+]] = OpTypeUntypedPointerKHR Function

; Test basic pointer parameters
; CHECK: OpFunctionParameter [[CROSS_PTR]]
; CHECK: OpFunctionParameter [[CROSS_PTR]]

define spir_kernel void @test_kernel(ptr addrspace(1) %input, ptr addrspace(1) %output) {
entry:
  %val = load i32, ptr addrspace(1) %input, align 4
  store i32 %val, ptr addrspace(1) %output, align 4
  ret void
}

; Test alloca produces untyped variable
; CHECK: OpFunction
; CHECK: [[ALLOCA:%[0-9]+]] = OpUntypedVariableKHR [[FUNC_PTR]] Function [[I32]]
define spir_kernel void @test_alloca() {
entry:
  %local = alloca i32, align 4
  store i32 42, ptr %local, align 4
  %val = load i32, ptr %local, align 4
  ret void
}

; Test GEP produces untyped access chain (PtrAccessChain for physical SPIR-V)
; CHECK: OpFunction
; CHECK: OpUntypedPtrAccessChainKHR [[CROSS_PTR]]
define spir_kernel void @test_gep(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr i32, ptr addrspace(1) %base, i64 5
  %val = load i32, ptr addrspace(1) %ptr, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}
