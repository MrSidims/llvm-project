; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.gfx942.o
; RUN: llvm-amdgpu-retarget --source=gfx942 --target=gfx90a %t.gfx942.o -o %t.gfx90a.o -v 2>&1 | FileCheck %s
; RUN: llvm-objdump -d %t.gfx90a.o 2>&1 | FileCheck %s --check-prefix=CHECK-DISASM

; Test that v_lshl_add_u64 is properly emulated when retargeting from gfx942 to gfx90a.
; gfx942 has v_lshl_add_u64, gfx90a does not and requires emulation with v_lshlrev_b64 + v_add.

; CHECK: Processing
; CHECK: Source architecture: gfx942
; CHECK: Target architecture: gfx90a
; CHECK: Disassembled
; CHECK: Liveness analysis:
; CHECK: Transformed

; Verify the disassembly contains the emulation sequence components
; The v_lshl_add_u64 is emulated as: v_lshlrev_b64 + v_add_co_u32 + v_addc_co_u32
; CHECK-DISASM: v_lshlrev_b64
; CHECK-DISASM: v_add_co_u32
; CHECK-DISASM: v_addc_co_u32

; Kernel that uses 64-bit address calculation with shift-add pattern
; Using vgpr inputs to force vector instructions
define amdgpu_kernel void @vector_scaled_access(ptr addrspace(1) %base, ptr addrspace(1) %indices, ptr addrspace(1) %out) {
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid64 = zext i32 %tid to i64

  ; Load a per-thread index from memory (ensures VGPR)
  %idx_ptr = getelementptr i64, ptr addrspace(1) %indices, i64 %tid64
  %idx = load i64, ptr addrspace(1) %idx_ptr

  ; Calculate scaled address: base + (idx << 3)
  ; This pattern should generate v_lshl_add_u64 on gfx942
  %shifted = shl i64 %idx, 3
  %base_int = ptrtoint ptr addrspace(1) %base to i64
  %addr = add i64 %base_int, %shifted

  ; Store result
  %out_ptr = getelementptr i64, ptr addrspace(1) %out, i64 %tid64
  store i64 %addr, ptr addrspace(1) %out_ptr
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
