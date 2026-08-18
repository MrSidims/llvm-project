; RUN: llc -mtriple=amdgcn < %s | FileCheck -check-prefix=DEFAULT %s
; RUN: llc -mtriple=amdgcn -amdgpu-v2s-copy-score-threshold=3 -amdgpu-v2s-copy-sv-copy-penalty=1 -amdgpu-v2s-copy-sibling-penalty=1 -amdgpu-v2s-copy-readfirstlane-penalty=1 < %s | FileCheck -check-prefix=EXPLICIT %s
; RUN: llc -mtriple=amdgcn -amdgpu-v2s-copy-score-threshold=0 < %s | FileCheck -check-prefix=THRESHOLD %s
; RUN: llc -mtriple=amdgcn -amdgpu-v2s-copy-sv-copy-penalty=8 -amdgpu-v2s-copy-sibling-penalty=8 -amdgpu-v2s-copy-readfirstlane-penalty=8 < %s | FileCheck -check-prefix=PENALTY %s

; DEFAULT-LABEL: udiv_i32:
; DEFAULT-COUNT-1: v_readfirstlane_b32
; DEFAULT-NOT:     v_readfirstlane_b32
; EXPLICIT-LABEL: udiv_i32:
; EXPLICIT-COUNT-1: v_readfirstlane_b32
; EXPLICIT-NOT:     v_readfirstlane_b32
; THRESHOLD-LABEL: udiv_i32:
; THRESHOLD-COUNT-3: v_readfirstlane_b32
; THRESHOLD-NOT:     v_readfirstlane_b32
; PENALTY-LABEL: udiv_i32:
; PENALTY-NOT:   v_readfirstlane_b32
define amdgpu_kernel void @udiv_i32(ptr addrspace(1) %out, i32 %x, i32 %y) {
  %r = udiv i32 %x, %y
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; DEFAULT-LABEL: udiv_i32_threshold:
; DEFAULT-COUNT-3: v_readfirstlane_b32
; DEFAULT-NOT:     v_readfirstlane_b32
; EXPLICIT-LABEL: udiv_i32_threshold:
; EXPLICIT-COUNT-1: v_readfirstlane_b32
; EXPLICIT-NOT:     v_readfirstlane_b32
; THRESHOLD-LABEL: udiv_i32_threshold:
; THRESHOLD-COUNT-3: v_readfirstlane_b32
; THRESHOLD-NOT:     v_readfirstlane_b32
; PENALTY-LABEL: udiv_i32_threshold:
; PENALTY-COUNT-3: v_readfirstlane_b32
; PENALTY-NOT:     v_readfirstlane_b32
define amdgpu_kernel void @udiv_i32_threshold(ptr addrspace(1) %out, i32 %x, i32 %y) #0 {
  %r = udiv i32 %x, %y
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; DEFAULT-LABEL: udiv_i32_sv_copy_penalty:
; DEFAULT-NOT:   v_readfirstlane_b32
; EXPLICIT-LABEL: udiv_i32_sv_copy_penalty:
; EXPLICIT-COUNT-1: v_readfirstlane_b32
; EXPLICIT-NOT:     v_readfirstlane_b32
; THRESHOLD-LABEL: udiv_i32_sv_copy_penalty:
; THRESHOLD-COUNT-3: v_readfirstlane_b32
; THRESHOLD-NOT:     v_readfirstlane_b32
; PENALTY-LABEL: udiv_i32_sv_copy_penalty:
; PENALTY-NOT:   v_readfirstlane_b32
define amdgpu_kernel void @udiv_i32_sv_copy_penalty(ptr addrspace(1) %out, i32 %x, i32 %y) #1 {
  %r = udiv i32 %x, %y
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; DEFAULT-LABEL: udiv_i32_sibling_penalty:
; DEFAULT-NOT:   v_readfirstlane_b32
; EXPLICIT-LABEL: udiv_i32_sibling_penalty:
; EXPLICIT-COUNT-1: v_readfirstlane_b32
; EXPLICIT-NOT:     v_readfirstlane_b32
; THRESHOLD-LABEL: udiv_i32_sibling_penalty:
; THRESHOLD-COUNT-3: v_readfirstlane_b32
; THRESHOLD-NOT:     v_readfirstlane_b32
; PENALTY-LABEL: udiv_i32_sibling_penalty:
; PENALTY-NOT:   v_readfirstlane_b32
define amdgpu_kernel void @udiv_i32_sibling_penalty(ptr addrspace(1) %out, i32 %x, i32 %y) #2 {
  %r = udiv i32 %x, %y
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; DEFAULT-LABEL: udiv_i32_readfirstlane_penalty:
; DEFAULT-NOT:   v_readfirstlane_b32
; EXPLICIT-LABEL: udiv_i32_readfirstlane_penalty:
; EXPLICIT-COUNT-1: v_readfirstlane_b32
; EXPLICIT-NOT:     v_readfirstlane_b32
; THRESHOLD-LABEL: udiv_i32_readfirstlane_penalty:
; THRESHOLD-COUNT-3: v_readfirstlane_b32
; THRESHOLD-NOT:     v_readfirstlane_b32
; PENALTY-LABEL: udiv_i32_readfirstlane_penalty:
; PENALTY-NOT:   v_readfirstlane_b32
define amdgpu_kernel void @udiv_i32_readfirstlane_penalty(ptr addrspace(1) %out, i32 %x, i32 %y) #3 {
  %r = udiv i32 %x, %y
  store i32 %r, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-v2s-copy-score-threshold"="0" }
attributes #1 = { "amdgpu-v2s-copy-sv-copy-penalty"="8" }
attributes #2 = { "amdgpu-v2s-copy-sibling-penalty"="8" }
attributes #3 = { "amdgpu-v2s-copy-readfirstlane-penalty"="8" }
