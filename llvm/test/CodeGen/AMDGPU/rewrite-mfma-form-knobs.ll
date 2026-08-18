; REQUIRES: asserts
; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -amdgpu-disable-rewrite-mfma-form-sched-stage=false -debug-only=machine-scheduler -o /dev/null %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -amdgpu-disable-rewrite-mfma-form-sched-stage=false -debug-only=machine-scheduler -amdgpu-rewrite-mfma-max-waves=1 -amdgpu-rewrite-mfma-archvgpr-trigger-percent=100 -amdgpu-rewrite-mfma-spill-multiplier=2 -amdgpu-rewrite-mfma-spill-balance=false -o /dev/null %s 2>&1 | FileCheck -check-prefix=EXPLICIT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -amdgpu-disable-rewrite-mfma-form-sched-stage=false -debug-only=machine-scheduler -amdgpu-rewrite-mfma-max-waves=4 -amdgpu-rewrite-mfma-archvgpr-trigger-percent=60 -amdgpu-rewrite-mfma-spill-multiplier=5 -amdgpu-rewrite-mfma-spill-balance=true -o /dev/null %s 2>&1 | FileCheck -check-prefix=FLAG %s

; A function attribute supplies the value of each knob.

; DEFAULT-LABEL: no_attrs:%bb.0
; DEFAULT: Rewrite MFMA form: MaxWaves = 1, ArchVGPRTriggerPercent = 100, SpillMultiplier = 2, SpillBalance = 0
; DEFAULT-LABEL: with_attrs:%bb.0
; DEFAULT: Rewrite MFMA form: MaxWaves = 2, ArchVGPRTriggerPercent = 75, SpillMultiplier = 3, SpillBalance = 1

; An explicitly given command line option wins over the attribute, even when it
; repeats the default.

; EXPLICIT-LABEL: no_attrs:%bb.0
; EXPLICIT: Rewrite MFMA form: MaxWaves = 1, ArchVGPRTriggerPercent = 100, SpillMultiplier = 2, SpillBalance = 0
; EXPLICIT-LABEL: with_attrs:%bb.0
; EXPLICIT: Rewrite MFMA form: MaxWaves = 1, ArchVGPRTriggerPercent = 100, SpillMultiplier = 2, SpillBalance = 0

; FLAG-LABEL: no_attrs:%bb.0
; FLAG: Rewrite MFMA form: MaxWaves = 4, ArchVGPRTriggerPercent = 60, SpillMultiplier = 5, SpillBalance = 1
; FLAG-LABEL: with_attrs:%bb.0
; FLAG: Rewrite MFMA form: MaxWaves = 4, ArchVGPRTriggerPercent = 60, SpillMultiplier = 5, SpillBalance = 1

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

attributes #0 = { "amdgpu-rewrite-mfma-max-waves"="2" "amdgpu-rewrite-mfma-archvgpr-trigger-percent"="75" "amdgpu-rewrite-mfma-spill-multiplier"="3" "amdgpu-rewrite-mfma-spill-balance"="true" }
