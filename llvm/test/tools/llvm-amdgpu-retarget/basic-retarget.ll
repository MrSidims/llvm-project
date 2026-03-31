; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.gfx942.o
; RUN: llvm-amdgpu-retarget --source=gfx942 --target=gfx90a %t.gfx942.o -o %t.gfx90a.o -v 2>&1 | FileCheck %s
; RUN: llvm-readelf -h %t.gfx90a.o | FileCheck %s --check-prefix=CHECK-ELF

; CHECK: Processing
; CHECK: Source architecture: gfx942
; CHECK: Target architecture: gfx90a
; CHECK: Processing .text section
; CHECK: Disassembled
; CHECK: Liveness analysis:
; CHECK: Transformed

; CHECK-ELF: Flags: 0x42

define amdgpu_kernel void @simple_add(ptr addrspace(1) %out, i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  store i32 %sum, ptr addrspace(1) %out
  ret void
}
