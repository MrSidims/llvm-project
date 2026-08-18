; REQUIRES: asserts

; Default build. The attributed kernel picks up its own values, the other one
; keeps the option defaults.
; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefix=DEFAULT %s

; Passing every knob its own default reproduces the default build wherever no
; attribute is present, and overrides the attribute where one is.
; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -amdgpu-sched-rp-error-margin=3 -amdgpu-sched-max-vgpr-pressure-inc=16 -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefix=EXPLICIT %s

; An explicit option always wins over the function attribute.
; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -amdgpu-sched-rp-error-margin=5 -amdgpu-sched-max-vgpr-pressure-inc=99 -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefix=FLAG %s

; DEFAULT-LABEL: no_attrs:%bb.0
; DEFAULT: ErrorMargin = 3, MaxVGPRPressureInc = 16
; DEFAULT-LABEL: with_attrs:%bb.0
; DEFAULT: ErrorMargin = 9, MaxVGPRPressureInc = 40

; EXPLICIT-LABEL: no_attrs:%bb.0
; EXPLICIT: ErrorMargin = 3, MaxVGPRPressureInc = 16
; EXPLICIT-LABEL: with_attrs:%bb.0
; EXPLICIT: ErrorMargin = 3, MaxVGPRPressureInc = 16

; FLAG-LABEL: no_attrs:%bb.0
; FLAG: ErrorMargin = 5, MaxVGPRPressureInc = 99
; FLAG-LABEL: with_attrs:%bb.0
; FLAG: ErrorMargin = 5, MaxVGPRPressureInc = 99

define amdgpu_kernel void @no_attrs(ptr addrspace(1) %out, i32 %n) {
entry:
  %v = load i32, ptr addrspace(1) %out
  %a = add i32 %v, %n
  store i32 %a, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @with_attrs(ptr addrspace(1) %out, i32 %n) #0 {
entry:
  %v = load i32, ptr addrspace(1) %out
  %a = add i32 %v, %n
  store i32 %a, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-sched-rp-error-margin"="9" "amdgpu-sched-max-vgpr-pressure-inc"="40" }
