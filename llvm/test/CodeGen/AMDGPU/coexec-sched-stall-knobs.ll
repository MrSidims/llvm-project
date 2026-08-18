; REQUIRES: asserts
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -amdgpu-sched-strategy=coexec -enable-post-misched=0 -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -amdgpu-sched-strategy=coexec -enable-post-misched=0 -debug-only=machine-scheduler -amdgpu-coexec-stall-weight-ready=1 -amdgpu-coexec-stall-weight-struct=1 -amdgpu-coexec-stall-weight-latency=1 -amdgpu-coexec-stall-combine=max -amdgpu-coexec-stall-slack=0 -o /dev/null %s 2>&1 | FileCheck -check-prefix=EXPLICIT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -amdgpu-sched-strategy=coexec -enable-post-misched=0 -debug-only=machine-scheduler -amdgpu-coexec-stall-weight-ready=5 -amdgpu-coexec-stall-weight-struct=6 -amdgpu-coexec-stall-weight-latency=7 -amdgpu-coexec-stall-combine=sum -amdgpu-coexec-stall-slack=9 -o /dev/null %s 2>&1 | FileCheck -check-prefix=FLAG %s

; A function attribute supplies the value of each knob.

; DEFAULT-LABEL: no_attrs:%bb.0
; DEFAULT: Stall weights: ready = 1, struct = 1, latency = 1, combine = max, slack = 0
; DEFAULT-LABEL: with_attrs:%bb.0
; DEFAULT: Stall weights: ready = 2, struct = 3, latency = 4, combine = sum, slack = 7

; An explicitly given command line option wins over the attribute, even when it
; repeats the default.

; EXPLICIT-LABEL: no_attrs:%bb.0
; EXPLICIT: Stall weights: ready = 1, struct = 1, latency = 1, combine = max, slack = 0
; EXPLICIT-LABEL: with_attrs:%bb.0
; EXPLICIT: Stall weights: ready = 1, struct = 1, latency = 1, combine = max, slack = 0

; FLAG-LABEL: no_attrs:%bb.0
; FLAG: Stall weights: ready = 5, struct = 6, latency = 7, combine = sum, slack = 9
; FLAG-LABEL: with_attrs:%bb.0
; FLAG: Stall weights: ready = 5, struct = 6, latency = 7, combine = sum, slack = 9

define amdgpu_kernel void @no_attrs(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %v = load float, ptr addrspace(1) %in
  %a = fadd float %v, 1.0
  store float %a, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @with_attrs(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %v = load float, ptr addrspace(1) %in
  %a = fadd float %v, 1.0
  store float %a, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-coexec-stall-weight-ready"="2" "amdgpu-coexec-stall-weight-struct"="3" "amdgpu-coexec-stall-weight-latency"="4" "amdgpu-coexec-stall-combine"="sum" "amdgpu-coexec-stall-slack"="7" }
