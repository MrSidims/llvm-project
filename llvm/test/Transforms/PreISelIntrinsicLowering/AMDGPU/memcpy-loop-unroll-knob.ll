; RUN: opt -mtriple=amdgpu12.00-amd-amdhsa -p=pre-isel-intrinsic-lowering -S < %s | FileCheck -check-prefixes=CHECK,DEFAULT %s

; An explicitly given option also overrides the attribute.
; RUN: opt -mtriple=amdgpu12.00-amd-amdhsa -p=pre-isel-intrinsic-lowering -S -amdgpu-memcpy-loop-unroll=16 < %s | FileCheck -check-prefixes=CHECK,EXPLICIT %s

; RUN: opt -mtriple=amdgpu12.00-amd-amdhsa -p=pre-isel-intrinsic-lowering -S -amdgpu-memcpy-loop-unroll=4 < %s | FileCheck -check-prefix=FLAG %s

; CHECK:    load <64 x i32>
; DEFAULT:  load <8 x i32>
; EXPLICIT: load <64 x i32>

; FLAG:     load <16 x i32>
; FLAG:     load <16 x i32>

define void @copy(ptr addrspace(1) %d, ptr addrspace(1) %s) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %d, ptr addrspace(1) %s, i64 4096, i1 false)
  ret void
}

define void @copy_attr(ptr addrspace(1) %d, ptr addrspace(1) %s) #0 {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %d, ptr addrspace(1) %s, i64 4096, i1 false)
  ret void
}

attributes #0 = { "amdgpu-memcpy-loop-unroll"="2" }
