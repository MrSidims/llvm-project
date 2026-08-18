; REQUIRES: asserts
; RUN: opt -mtriple=amdgpu-unknown-amdhsa -passes=loop-unroll -S -debug-only=AMDGPUtti < %s -o /dev/null 2>&1 | FileCheck -check-prefixes=CHECK,DEFAULT %s

; An explicitly given option also overrides the attribute.
; RUN: opt -mtriple=amdgpu-unknown-amdhsa -passes=loop-unroll -S -debug-only=AMDGPUtti \
; RUN:   -amdgpu-unroll-threshold-private=2700 -amdgpu-unroll-threshold-local=1000 \
; RUN:   -amdgpu-unroll-threshold-if=200 < %s -o /dev/null 2>&1 | FileCheck -check-prefixes=CHECK,EXPLICIT %s

; RUN: opt -mtriple=amdgpu-unknown-amdhsa -passes=loop-unroll -S -debug-only=AMDGPUtti \
; RUN:   -amdgpu-unroll-threshold-private=900 -amdgpu-unroll-threshold-local=800 \
; RUN:   -amdgpu-unroll-threshold-if=25 < %s -o /dev/null 2>&1 | FileCheck -check-prefix=FLAG %s

; CHECK:    Set unroll threshold 2700 for loop:
; CHECK:    Set unroll threshold 1000 for loop:
; CHECK:    Set unroll threshold 500 for loop:
; DEFAULT:  Set unroll threshold 1500 for loop:
; DEFAULT:  Set unroll threshold 600 for loop:
; DEFAULT:  Set unroll threshold 350 for loop:
; EXPLICIT: Set unroll threshold 2700 for loop:
; EXPLICIT: Set unroll threshold 1000 for loop:
; EXPLICIT: Set unroll threshold 500 for loop:

; FLAG:     Set unroll threshold 900 for loop:
; FLAG:     Set unroll threshold 800 for loop:
; FLAG:     Set unroll threshold 325 for loop:
; FLAG:     Set unroll threshold 900 for loop:
; FLAG:     Set unroll threshold 800 for loop:
; FLAG:     Set unroll threshold 325 for loop:

@lds = internal unnamed_addr addrspace(3) global [64 x i32] poison

define amdgpu_kernel void @unroll_private(ptr addrspace(1) nocapture %a) {
entry:
  %arr = alloca [64 x i32], align 4, addrspace(5)
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %a, i32 %i
  %v = load i32, ptr addrspace(1) %arrayidx, align 4
  %rem = srem i32 %i, 64
  %arrayidx3 = getelementptr inbounds [64 x i32], ptr addrspace(5) %arr, i32 0, i32 %rem
  store i32 %v, ptr addrspace(5) %arrayidx3, align 4
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define amdgpu_kernel void @unroll_local(ptr addrspace(1) nocapture %a) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %a, i32 %i
  %v = load i32, ptr addrspace(1) %arrayidx, align 4
  %rem = srem i32 %i, 64
  %arrayidx3 = getelementptr inbounds [64 x i32], ptr addrspace(3) @lds, i32 0, i32 %rem
  store i32 %v, ptr addrspace(3) %arrayidx3, align 4
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define amdgpu_kernel void @unroll_if(ptr addrspace(1) nocapture %a) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %if.end ]
  %cond = icmp ult i32 %i, 7
  br i1 %cond, label %if.then, label %if.end

if.then:
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %a, i32 %i
  store i32 %i, ptr addrspace(1) %arrayidx, align 4
  br label %if.end

if.end:
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define amdgpu_kernel void @unroll_private_attr(ptr addrspace(1) nocapture %a) #0 {
entry:
  %arr = alloca [64 x i32], align 4, addrspace(5)
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %a, i32 %i
  %v = load i32, ptr addrspace(1) %arrayidx, align 4
  %rem = srem i32 %i, 64
  %arrayidx3 = getelementptr inbounds [64 x i32], ptr addrspace(5) %arr, i32 0, i32 %rem
  store i32 %v, ptr addrspace(5) %arrayidx3, align 4
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define amdgpu_kernel void @unroll_local_attr(ptr addrspace(1) nocapture %a) #1 {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %a, i32 %i
  %v = load i32, ptr addrspace(1) %arrayidx, align 4
  %rem = srem i32 %i, 64
  %arrayidx3 = getelementptr inbounds [64 x i32], ptr addrspace(3) @lds, i32 0, i32 %rem
  store i32 %v, ptr addrspace(3) %arrayidx3, align 4
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define amdgpu_kernel void @unroll_if_attr(ptr addrspace(1) nocapture %a) #2 {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %if.end ]
  %cond = icmp ult i32 %i, 7
  br i1 %cond, label %if.then, label %if.end

if.then:
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %a, i32 %i
  store i32 %i, ptr addrspace(1) %arrayidx, align 4
  br label %if.end

if.end:
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

attributes #0 = { "amdgpu-unroll-threshold-private"="1500" }
attributes #1 = { "amdgpu-unroll-threshold-local"="600" }
attributes #2 = { "amdgpu-unroll-threshold-if"="50" }
