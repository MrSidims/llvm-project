; RUN: opt -mtriple=amdgpu9.0a-amd-amdhsa -passes=loop-vectorize -S < %s | FileCheck -check-prefixes=CHECK,DEFAULT %s

; An explicitly given option also overrides the attribute.
; RUN: opt -mtriple=amdgpu9.0a-amd-amdhsa -passes=loop-vectorize -S -amdgpu-max-interleave-factor=8 < %s | FileCheck -check-prefixes=CHECK,EXPLICIT %s

; RUN: opt -mtriple=amdgpu9.0a-amd-amdhsa -passes=loop-vectorize -S -amdgpu-max-interleave-factor=1 < %s | FileCheck -check-prefix=FLAG %s

; CHECK-LABEL: @interleave(
; CHECK-COUNT-2: load <2 x float>
; CHECK-NOT: load <2 x float>

; DEFAULT-LABEL: @interleave_attr(
; DEFAULT: load <2 x float>
; DEFAULT-NOT: load <2 x float>

; EXPLICIT-LABEL: @interleave_attr(
; EXPLICIT-COUNT-2: load <2 x float>
; EXPLICIT-NOT: load <2 x float>

; FLAG-LABEL: @interleave(
; FLAG: load <2 x float>
; FLAG-NOT: load <2 x float>

; FLAG-LABEL: @interleave_attr(
; FLAG: load <2 x float>
; FLAG-NOT: load <2 x float>

define float @interleave(ptr addrspace(1) noalias %s) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %q.04 = phi float [ 0.0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %s, i64 %indvars.iv
  %load = load float, ptr addrspace(1) %arrayidx, align 4
  %add = fadd fast float %q.04, %load
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  %add.lcssa = phi float [ %add, %for.body ]
  ret float %add.lcssa
}

define float @interleave_attr(ptr addrspace(1) noalias %s) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %q.04 = phi float [ 0.0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %s, i64 %indvars.iv
  %load = load float, ptr addrspace(1) %arrayidx, align 4
  %add = fadd fast float %q.04, %load
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  %add.lcssa = phi float [ %add, %for.body ]
  ret float %add.lcssa
}

attributes #0 = { "amdgpu-max-interleave-factor"="1" }
