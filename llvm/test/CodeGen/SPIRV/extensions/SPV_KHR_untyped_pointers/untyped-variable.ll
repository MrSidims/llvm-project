; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Test OpUntypedVariableKHR for local variables (alloca).

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: [[I8:%[0-9]+]] = OpTypeInt 8 0
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[I64:%[0-9]+]] = OpTypeInt 64 0
; CHECK-DAG: [[F32:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[F64:%[0-9]+]] = OpTypeFloat 64
; CHECK-DAG: [[FUNC_PTR:%[0-9]+]] = OpTypeUntypedPointerKHR Function

; Test i8 alloca - should use OpUntypedVariableKHR with i8 data type
; CHECK: OpFunction
; CHECK: OpUntypedVariableKHR [[FUNC_PTR]] Function [[I8]]
define spir_kernel void @test_alloca_i8() {
entry:
  %local = alloca i8, align 1
  store i8 42, ptr %local, align 1
  ret void
}

; Test i32 alloca - should use OpUntypedVariableKHR with i32 data type
; CHECK: OpFunction
; CHECK: OpUntypedVariableKHR [[FUNC_PTR]] Function [[I32]]
define spir_kernel void @test_alloca_i32() {
entry:
  %local = alloca i32, align 4
  store i32 42, ptr %local, align 4
  ret void
}

; Test i64 alloca - should use OpUntypedVariableKHR with i64 data type
; CHECK: OpFunction
; CHECK: OpUntypedVariableKHR [[FUNC_PTR]] Function [[I64]]
define spir_kernel void @test_alloca_i64() {
entry:
  %local = alloca i64, align 8
  store i64 42, ptr %local, align 8
  ret void
}

; Test float alloca - should use OpUntypedVariableKHR with float data type
; CHECK: OpFunction
; CHECK: OpUntypedVariableKHR [[FUNC_PTR]] Function [[F32]]
define spir_kernel void @test_alloca_float() {
entry:
  %local = alloca float, align 4
  store float 3.14, ptr %local, align 4
  ret void
}

; Test double alloca - should use OpUntypedVariableKHR with double data type
; CHECK: OpFunction
; CHECK: OpUntypedVariableKHR [[FUNC_PTR]] Function [[F64]]
define spir_kernel void @test_alloca_double() {
entry:
  %local = alloca double, align 8
  store double 3.14, ptr %local, align 8
  ret void
}

; Test multiple allocas in the same function
; CHECK: OpFunction
; CHECK-DAG: OpUntypedVariableKHR [[FUNC_PTR]] Function [[I32]]
; CHECK-DAG: OpUntypedVariableKHR [[FUNC_PTR]] Function [[F32]]
define spir_kernel void @test_multiple_alloca() {
entry:
  %int_local = alloca i32, align 4
  %float_local = alloca float, align 4
  store i32 42, ptr %int_local, align 4
  store float 3.14, ptr %float_local, align 4
  ret void
}

; Test alloca with load and store
; CHECK: OpFunction
; CHECK: [[VAR:%[0-9]+]] = OpUntypedVariableKHR [[FUNC_PTR]] Function [[I32]]
; CHECK: OpStore [[VAR]]
; CHECK: OpLoad [[I32]] [[VAR]]
define spir_kernel void @test_alloca_load_store(ptr addrspace(1) %out) {
entry:
  %local = alloca i32, align 4
  store i32 100, ptr %local, align 4
  %val = load i32, ptr %local, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}
