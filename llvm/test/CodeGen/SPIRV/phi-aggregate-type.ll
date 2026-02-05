; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; Check that PHI nodes and select instructions with aggregate (struct) types
; are scalarized by SPIRVPrepareFunctions into per-field scalar PHI/select
; operations. Verify that the generated SPIR-V uses scalar OpPhi/OpSelect
; on element types and OpCompositeInsert to reconstruct the aggregate.
;
; Functions are ordered to match SPIR-V output: void-returning (non-cloned)
; functions appear before aggregate-returning (cloned) functions.

; CHECK-DAG: %[[#Long:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#Long]] %[[#Long]]
; CHECK-DAG: %[[#InnerStruct:]] = OpTypeStruct %[[#Int]] %[[#Int]]
; CHECK-DAG: %[[#NestedStruct:]] = OpTypeStruct %[[#InnerStruct]] %[[#Long]]

; PHI with result used via extractvalue (not returned directly).
; CHECK: OpFunction
; CHECK: OpPhi %[[#Long]]
; CHECK-NEXT: OpPhi %[[#Long]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpCompositeExtract %[[#Long]]
; CHECK: OpStore
define spir_func void @test_phi_extractvalue(i1 %cond, ptr addrspace(4) %out) addrspace(4) {
entry:
  br i1 %cond, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  %p = phi { i64, i64 } [ zeroinitializer, %then ], [ { i64 1, i64 2 }, %else ]
  %v = extractvalue { i64, i64 } %p, 0
  store i64 %v, ptr addrspace(4) %out
  ret void
}

; Select with aggregate constant operands, result used via extractvalue.
; CHECK: OpFunction
; CHECK: OpSelect %[[#Long]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpSelect %[[#Long]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpCompositeExtract %[[#Long]]
; CHECK: OpStore
define spir_func void @test_select_extractvalue(i1 %cond, ptr addrspace(4) %out) addrspace(4) {
entry:
  %s = select i1 %cond, { i64, i64 } zeroinitializer, { i64, i64 } { i64 1, i64 2 }
  %v = extractvalue { i64, i64 } %s, 0
  store i64 %v, ptr addrspace(4) %out
  ret void
}

; PHI with a non-constant incoming value (from a load).
; CHECK: OpFunction
; CHECK: OpLoad %[[#Struct]]
; CHECK: OpCompositeExtract %[[#Long]]
; CHECK: OpCompositeExtract %[[#Long]]
; CHECK: OpPhi %[[#Long]]
; CHECK-NEXT: OpPhi %[[#Long]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpCompositeExtract %[[#Long]]
; CHECK: OpStore
define spir_func void @test_phi_non_constant(i1 %cond, ptr addrspace(4) %in, ptr addrspace(4) %out) addrspace(4) {
entry:
  %val = load { i64, i64 }, ptr addrspace(4) %in
  br i1 %cond, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  %p = phi { i64, i64 } [ %val, %then ], [ zeroinitializer, %else ]
  %v = extractvalue { i64, i64 } %p, 0
  store i64 %v, ptr addrspace(4) %out
  ret void
}

; Loop-carried aggregate PHI.
; CHECK: OpFunction
; CHECK: OpPhi %[[#Int]]
; CHECK: OpPhi %[[#Long]]
; CHECK: OpPhi %[[#Long]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpBranchConditional
define spir_func void @test_phi_loop(i32 %n, ptr addrspace(4) %out) addrspace(4) {
entry:
  br label %loop

loop:
  %acc = phi { i64, i64 } [ zeroinitializer, %entry ], [ %updated, %loop ]
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %f0 = extractvalue { i64, i64 } %acc, 0
  %f1 = extractvalue { i64, i64 } %acc, 1
  %ext = zext i32 %i to i64
  %new0 = add i64 %f0, %ext
  %new1 = add i64 %f1, 1
  %tmp = insertvalue { i64, i64 } poison, i64 %new0, 0
  %updated = insertvalue { i64, i64 } %tmp, i64 %new1, 1
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  %v = extractvalue { i64, i64 } %acc, 0
  store i64 %v, ptr addrspace(4) %out
  ret void
}

; PHI with two constant incoming values, result returned directly.
; CHECK: OpFunction %[[#Struct]]
; CHECK: OpPhi %[[#Long]]
; CHECK-NEXT: OpPhi %[[#Long]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpReturnValue
define spir_func { i64, i64 } @test_phi_ret(i1 %cond) addrspace(4) {
entry:
  br i1 %cond, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  %p = phi { i64, i64 } [ zeroinitializer, %then ], [ { i64 1, i64 2 }, %else ]
  ret { i64, i64 } %p
}

; PHI with three incoming blocks (more predecessors than struct fields).
; CHECK: OpFunction %[[#Struct]]
; CHECK: OpPhi %[[#Long]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK-NEXT: OpPhi %[[#Long]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpReturnValue
define spir_func { i64, i64 } @test_phi_three_preds(i32 %sel) addrspace(4) {
entry:
  switch i32 %sel, label %bb3 [
    i32 0, label %bb1
    i32 1, label %bb2
  ]

bb1:
  br label %merge

bb2:
  br label %merge

bb3:
  br label %merge

merge:
  %p = phi { i64, i64 } [ zeroinitializer, %bb1 ], [ { i64 1, i64 2 }, %bb2 ], [ { i64 3, i64 4 }, %bb3 ]
  ret { i64, i64 } %p
}

; Select with aggregate constant operands, result returned directly.
; CHECK: OpFunction %[[#Struct]]
; CHECK: OpSelect %[[#Long]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpSelect %[[#Long]]
; CHECK: OpCompositeInsert %[[#Struct]]
; CHECK: OpReturnValue
define spir_func { i64, i64 } @test_select_ret(i1 %cond) addrspace(4) {
entry:
  %s = select i1 %cond, { i64, i64 } zeroinitializer, { i64, i64 } { i64 1, i64 2 }
  ret { i64, i64 } %s
}

; Nested aggregate type — requires multiple scalarization rounds.
; The outer struct { { i32, i32 }, i64 } produces three scalar PHIs.
; CHECK: OpFunction %[[#NestedStruct]]
; CHECK: OpPhi %[[#Long]]
; CHECK-NEXT: OpPhi %[[#Int]]
; CHECK-NEXT: OpPhi %[[#Int]]
; CHECK: OpCompositeInsert %[[#InnerStruct]]
; CHECK: OpCompositeInsert %[[#InnerStruct]]
; CHECK: OpCompositeInsert %[[#NestedStruct]]
; CHECK: OpCompositeInsert %[[#NestedStruct]]
; CHECK: OpReturnValue
define spir_func { { i32, i32 }, i64 } @test_phi_nested(i1 %cond) addrspace(4) {
entry:
  br i1 %cond, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  %p = phi { { i32, i32 }, i64 } [ zeroinitializer, %then ], [ { { i32, i32 } { i32 1, i32 2 }, i64 3 }, %else ]
  ret { { i32, i32 }, i64 } %p
}
