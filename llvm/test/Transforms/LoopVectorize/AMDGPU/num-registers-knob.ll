; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -passes=loop-vectorize -S < %s | FileCheck -check-prefixes=CHECK,DEFAULT %s

; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -passes=loop-vectorize -S -amdgpu-num-registers=4 < %s | FileCheck -check-prefixes=CHECK,EXPLICIT %s

; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -passes=loop-vectorize -S -amdgpu-num-registers=64 < %s | FileCheck -check-prefixes=CHECK,FLAG %s

; The register count and not getMaxInterleaveFactor is what caps interleaving
; here, so raising it reaches the factor of 8 the target already allows.

; CHECK-LABEL: @interleave(
; DEFAULT-COUNT-2:  load <2 x float>
; DEFAULT-NOT:      load <2 x float>
; EXPLICIT-COUNT-2: load <2 x float>
; EXPLICIT-NOT:     load <2 x float>
; FLAG-COUNT-8:     load <2 x float>
; FLAG-NOT:         load <2 x float>
define float @interleave(ptr addrspace(1) noalias %in, i32 %n) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %add, %loop ]
  %gep = getelementptr inbounds float, ptr addrspace(1) %in, i32 %iv
  %val = load float, ptr addrspace(1) %gep, align 4
  %add = fadd fast float %acc, %val
  %iv.next = add nuw nsw i32 %iv, 1
  %cmp = icmp eq i32 %iv.next, 1024
  br i1 %cmp, label %exit, label %loop

exit:
  ret float %add
}
