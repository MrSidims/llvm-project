; REQUIRES: asserts
; RUN: llc -mtriple=amdgpu9.42-amd-amdhsa -debug-only=igrouplp -o /dev/null %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
; RUN: llc -mtriple=amdgpu9.42-amd-amdhsa -debug-only=igrouplp -amdgpu-igrouplp-small-gemm-rounds=3 -amdgpu-igrouplp-small-gemm-ds-size=2 -amdgpu-igrouplp-small-gemm-mfma-size=1 -amdgpu-igrouplp-exp-small-mfma-enablement=2 -amdgpu-igrouplp-exp-small-exp-requirement=4 -amdgpu-igrouplp-exp-small-trans-count=32 -amdgpu-igrouplp-exp-large-mfma-enablement=4 -amdgpu-igrouplp-exp-large-exp-requirement=4 -amdgpu-igrouplp-exp-large-trans-count=64 -o /dev/null %s 2>&1 | FileCheck -check-prefix=EXPLICIT %s
; RUN: llc -mtriple=amdgpu9.42-amd-amdhsa -debug-only=igrouplp -amdgpu-igrouplp-small-gemm-rounds=6 -amdgpu-igrouplp-exp-large-trans-count=96 -o /dev/null %s 2>&1 | FileCheck -check-prefix=FLAG %s

; DEFAULT: Small gemm: Rounds = 3, DSSize = 2, MFMASize = 1
; DEFAULT: Small gemm: Rounds = 5, DSSize = 4, MFMASize = 3
; DEFAULT: Exp interleave: measured 1/4/8, small 2/4/32, large 4/4/64
; DEFAULT: Exp interleave: measured 1/4/8, small 1/2/24, large 6/7/48

; EXPLICIT: Small gemm: Rounds = 3, DSSize = 2, MFMASize = 1
; EXPLICIT: Small gemm: Rounds = 3, DSSize = 2, MFMASize = 1
; EXPLICIT: Exp interleave: measured 1/4/8, small 2/4/32, large 4/4/64
; EXPLICIT: Exp interleave: measured 1/4/8, small 2/4/32, large 4/4/64

; FLAG: Small gemm: Rounds = 6, DSSize = 2, MFMASize = 1
; FLAG: Small gemm: Rounds = 6, DSSize = 4, MFMASize = 3
; FLAG: Exp interleave: measured 1/4/8, small 2/4/32, large 4/4/96
; FLAG: Exp interleave: measured 1/4/8, small 1/2/24, large 6/7/96

define amdgpu_kernel void @small_gemm(ptr addrspace(3) noalias %in, ptr addrspace(3) noalias %out) {
  call void @llvm.amdgcn.iglp.opt(i32 0)
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %load.0.addr = getelementptr <32 x float>, ptr addrspace(3) %in, i32 %idx
  %load.0 = load <32 x float>, ptr addrspace(3) %load.0.addr
  %load.1.addr = getelementptr <32 x float>, ptr addrspace(3) %load.0.addr, i32 64
  %load.1 = load <32 x float>, ptr addrspace(3) %load.1.addr
  %mai.0 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.0, i32 0, i32 0, i32 0)
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.1, i32 0, i32 0, i32 0)
  %store.0.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 %idx
  store <32 x float> %mai.0, ptr addrspace(3) %store.0.addr
  %store.1.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 64
  store <32 x float> %mai.1, ptr addrspace(3) %store.1.addr
  ret void
}

define amdgpu_kernel void @small_gemm_attrs(ptr addrspace(3) noalias %in, ptr addrspace(3) noalias %out) #0 {
  call void @llvm.amdgcn.iglp.opt(i32 0)
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %load.0.addr = getelementptr <32 x float>, ptr addrspace(3) %in, i32 %idx
  %load.0 = load <32 x float>, ptr addrspace(3) %load.0.addr
  %load.1.addr = getelementptr <32 x float>, ptr addrspace(3) %load.0.addr, i32 64
  %load.1 = load <32 x float>, ptr addrspace(3) %load.1.addr
  %mai.0 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.0, i32 0, i32 0, i32 0)
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.1, i32 0, i32 0, i32 0)
  %store.0.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 %idx
  store <32 x float> %mai.0, ptr addrspace(3) %store.0.addr
  %store.1.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 64
  store <32 x float> %mai.1, ptr addrspace(3) %store.1.addr
  ret void
}

define amdgpu_kernel void @exp_pipe(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  call void @llvm.amdgcn.iglp.opt(i32 2)
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr float, ptr addrspace(1) %in, i32 %id
  %c = load <16 x float>, ptr addrspace(1) %out
  %v0.0 = load volatile float, ptr addrspace(1) %gep
  %v0.1 = load volatile float, ptr addrspace(1) %gep
  %v0.2 = load volatile float, ptr addrspace(1) %gep
  %v0.3 = load volatile float, ptr addrspace(1) %gep
  %e0.0 = call float @llvm.exp2.f32(float %v0.0)
  %e0.1 = call float @llvm.exp2.f32(float %v0.1)
  %e0.2 = call float @llvm.exp2.f32(float %v0.2)
  %e0.3 = call float @llvm.exp2.f32(float %v0.3)
  %h0.0 = fptrunc float %e0.0 to half
  %h0.1 = fptrunc float %e0.1 to half
  %h0.2 = fptrunc float %e0.2 to half
  %h0.3 = fptrunc float %e0.3 to half
  %i0.0 = insertelement <4 x half> poison, half %h0.0, i32 0
  %i0.1 = insertelement <4 x half> %i0.0, half %h0.1, i32 1
  %i0.2 = insertelement <4 x half> %i0.1, half %h0.2, i32 2
  %i0.3 = insertelement <4 x half> %i0.2, half %h0.3, i32 3
  %v1.0 = load volatile float, ptr addrspace(1) %gep
  %v1.1 = load volatile float, ptr addrspace(1) %gep
  %v1.2 = load volatile float, ptr addrspace(1) %gep
  %v1.3 = load volatile float, ptr addrspace(1) %gep
  %e1.0 = call float @llvm.exp2.f32(float %v1.0)
  %e1.1 = call float @llvm.exp2.f32(float %v1.1)
  %e1.2 = call float @llvm.exp2.f32(float %v1.2)
  %e1.3 = call float @llvm.exp2.f32(float %v1.3)
  %h1.0 = fptrunc float %e1.0 to half
  %h1.1 = fptrunc float %e1.1 to half
  %h1.2 = fptrunc float %e1.2 to half
  %h1.3 = fptrunc float %e1.3 to half
  %i1.0 = insertelement <4 x half> poison, half %h1.0, i32 0
  %i1.1 = insertelement <4 x half> %i1.0, half %h1.1, i32 1
  %i1.2 = insertelement <4 x half> %i1.1, half %h1.2, i32 2
  %i1.3 = insertelement <4 x half> %i1.2, half %h1.3, i32 3
  %m0 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> %i0.3, <4 x half> %i0.3, <16 x float> %c, i32 0, i32 0, i32 0)
  %m1 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> %i1.3, <4 x half> %i1.3, <16 x float> %m0, i32 0, i32 0, i32 0)
  store <16 x float> %m1, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @exp_pipe_attrs(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 {
  call void @llvm.amdgcn.iglp.opt(i32 2)
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr float, ptr addrspace(1) %in, i32 %id
  %c = load <16 x float>, ptr addrspace(1) %out
  %v0.0 = load volatile float, ptr addrspace(1) %gep
  %v0.1 = load volatile float, ptr addrspace(1) %gep
  %v0.2 = load volatile float, ptr addrspace(1) %gep
  %v0.3 = load volatile float, ptr addrspace(1) %gep
  %e0.0 = call float @llvm.exp2.f32(float %v0.0)
  %e0.1 = call float @llvm.exp2.f32(float %v0.1)
  %e0.2 = call float @llvm.exp2.f32(float %v0.2)
  %e0.3 = call float @llvm.exp2.f32(float %v0.3)
  %h0.0 = fptrunc float %e0.0 to half
  %h0.1 = fptrunc float %e0.1 to half
  %h0.2 = fptrunc float %e0.2 to half
  %h0.3 = fptrunc float %e0.3 to half
  %i0.0 = insertelement <4 x half> poison, half %h0.0, i32 0
  %i0.1 = insertelement <4 x half> %i0.0, half %h0.1, i32 1
  %i0.2 = insertelement <4 x half> %i0.1, half %h0.2, i32 2
  %i0.3 = insertelement <4 x half> %i0.2, half %h0.3, i32 3
  %v1.0 = load volatile float, ptr addrspace(1) %gep
  %v1.1 = load volatile float, ptr addrspace(1) %gep
  %v1.2 = load volatile float, ptr addrspace(1) %gep
  %v1.3 = load volatile float, ptr addrspace(1) %gep
  %e1.0 = call float @llvm.exp2.f32(float %v1.0)
  %e1.1 = call float @llvm.exp2.f32(float %v1.1)
  %e1.2 = call float @llvm.exp2.f32(float %v1.2)
  %e1.3 = call float @llvm.exp2.f32(float %v1.3)
  %h1.0 = fptrunc float %e1.0 to half
  %h1.1 = fptrunc float %e1.1 to half
  %h1.2 = fptrunc float %e1.2 to half
  %h1.3 = fptrunc float %e1.3 to half
  %i1.0 = insertelement <4 x half> poison, half %h1.0, i32 0
  %i1.1 = insertelement <4 x half> %i1.0, half %h1.1, i32 1
  %i1.2 = insertelement <4 x half> %i1.1, half %h1.2, i32 2
  %i1.3 = insertelement <4 x half> %i1.2, half %h1.3, i32 3
  %m0 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> %i0.3, <4 x half> %i0.3, <16 x float> %c, i32 0, i32 0, i32 0)
  %m1 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> %i1.3, <4 x half> %i1.3, <16 x float> %m0, i32 0, i32 0, i32 0)
  store <16 x float> %m1, ptr addrspace(1) %out
  ret void
}

declare void @llvm.amdgcn.iglp.opt(i32)

attributes #0 = {
  "amdgpu-igrouplp-small-gemm-rounds"="5"
  "amdgpu-igrouplp-small-gemm-ds-size"="4"
  "amdgpu-igrouplp-small-gemm-mfma-size"="3"
  "amdgpu-igrouplp-exp-small-mfma-enablement"="1"
  "amdgpu-igrouplp-exp-small-exp-requirement"="2"
  "amdgpu-igrouplp-exp-small-trans-count"="24"
  "amdgpu-igrouplp-exp-large-mfma-enablement"="6"
  "amdgpu-igrouplp-exp-large-exp-requirement"="7"
  "amdgpu-igrouplp-exp-large-trans-count"="48" }
