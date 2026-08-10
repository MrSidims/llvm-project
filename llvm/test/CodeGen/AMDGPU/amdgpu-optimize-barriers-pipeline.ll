; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -O1 -amdgpu-enable-optimize-barriers -debug-pass=Structure -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=LEGACY
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -O1 -amdgpu-enable-optimize-barriers -enable-new-pm -print-pipeline-passes=tree < %s 2>&1 | FileCheck %s --check-prefix=NPM

; The pass must run before AMDGPU lower intrinsics which splits s_barrier
; into signal and wait halves on gfx12.

; LEGACY: AMDGPU Optimize Barriers
; LEGACY: AMDGPU lower intrinsics

; NPM: amdgpu-optimize-barriers
; NPM: amdgpu-lower-intrinsics

define amdgpu_kernel void @empty() {
  ret void
}
