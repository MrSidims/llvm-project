; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; Check that select on aggregate (struct/array) types does not crash the
; compiler. SPIRVPrepareFunctions scalarizes aggregate selects into per-field
; extractvalue + scalar select + insertvalue chains before later passes see
; them.

; CHECK-DAG: %[[#Long:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#Long]] %[[#Long]]

; Struct select with zeroinitializer (unused result — optimized away).
; CHECK: OpFunction
; CHECK-NOT: OpSelect
; CHECK: OpReturn
; CHECK: OpFunctionEnd
define spir_kernel void @test_select_struct_dead() addrspace(4) {
entry:
  %s = select i1 false, { i64, i64 } zeroinitializer, { i64, i64 } zeroinitializer
  ret void
}

; Struct select with used result via extractvalue.
; CHECK: OpFunction
; CHECK: OpSelect %[[#Long]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpSelect %[[#Long]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpCompositeExtract %[[#Long]]
; CHECK: OpStore
define spir_kernel void @test_select_struct_used(i1 %cond) addrspace(4) {
entry:
  %s = select i1 %cond, { i64, i64 } zeroinitializer, { i64, i64 } zeroinitializer
  %v = extractvalue { i64, i64 } %s, 0
  store i64 %v, ptr addrspace(1) null
  ret void
}

; Array select with zeroinitializer (unused result — optimized away).
; CHECK: OpFunction
; CHECK-NOT: OpSelect
; CHECK: OpReturn
; CHECK: OpFunctionEnd
define spir_kernel void @test_select_array_dead() addrspace(4) {
entry:
  %a = select i1 false, [2 x i64] zeroinitializer, [2 x i64] zeroinitializer
  ret void
}
