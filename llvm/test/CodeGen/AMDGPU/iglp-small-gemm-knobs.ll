; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck -check-prefix=DEFAULT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -amdgpu-igrouplp-small-gemm-rounds=3 -amdgpu-igrouplp-small-gemm-ds-size=2 -amdgpu-igrouplp-small-gemm-mfma-size=1 < %s | FileCheck -check-prefix=EXPLICIT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -amdgpu-igrouplp-small-gemm-rounds=6 < %s | FileCheck -check-prefix=ROUNDS %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -amdgpu-igrouplp-small-gemm-ds-size=4 < %s | FileCheck -check-prefix=DSSIZE %s

; DEFAULT: test_iglp_opt_mfma_gemm:
; DEFAULT: TotalNumVgprs: 68
; DEFAULT: Occupancy: 7
; DEFAULT: test_iglp_opt_mfma_gemm_attr:
; DEFAULT: TotalNumVgprs: 163
; DEFAULT: Occupancy: 3

; EXPLICIT: test_iglp_opt_mfma_gemm:
; EXPLICIT: TotalNumVgprs: 68
; EXPLICIT: Occupancy: 7
; EXPLICIT: test_iglp_opt_mfma_gemm_attr:
; EXPLICIT: TotalNumVgprs: 68
; EXPLICIT: Occupancy: 7

; ROUNDS: test_iglp_opt_mfma_gemm:
; ROUNDS: TotalNumVgprs: 163
; ROUNDS: Occupancy: 3
; ROUNDS: test_iglp_opt_mfma_gemm_attr:
; ROUNDS: TotalNumVgprs: 163
; ROUNDS: Occupancy: 3

; DSSIZE: test_iglp_opt_mfma_gemm:
; DSSIZE: TotalNumVgprs: 163
; DSSIZE: Occupancy: 3
; DSSIZE: test_iglp_opt_mfma_gemm_attr:
; DSSIZE: TotalNumVgprs: 163
; DSSIZE: Occupancy: 3

define amdgpu_kernel void @test_iglp_opt_mfma_gemm(ptr addrspace(3) noalias %in, ptr addrspace(3) noalias %out) #0 {
entry:
  call void @llvm.amdgcn.iglp.opt(i32 0)
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %load.0.addr = getelementptr <32 x float>, ptr addrspace(3) %in, i32 %idx
  %load.0 = load <32 x float>, ptr addrspace(3) %load.0.addr
  %load.1.addr = getelementptr <32 x float>, ptr addrspace(3) %load.0.addr, i32 64
  %load.1 = load <32 x float>, ptr addrspace(3) %load.1.addr
  %load.2.addr = getelementptr <32 x float>, ptr addrspace(3) %load.1.addr, i32 128
  %load.2 = load <32 x float>, ptr addrspace(3) %load.2.addr
  %load.3.addr = getelementptr <32 x float>, ptr addrspace(3) %load.2.addr, i32 192
  %load.3 = load <32 x float>, ptr addrspace(3) %load.3.addr
  %load.4.addr = getelementptr <32 x float>, ptr addrspace(3) %load.3.addr, i32 256
  %load.4 = load <32 x float>, ptr addrspace(3) %load.4.addr
  %mai.0 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.0, i32 0, i32 0, i32 0)
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.2, i32 0, i32 0, i32 0)
  %mai.3 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.3, i32 0, i32 0, i32 0)
  %mai.4 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.4, i32 0, i32 0, i32 0)
  %store.0.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 %idx
  store <32 x float> %mai.0, ptr addrspace(3) %store.0.addr
  %store.1.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 64
  store <32 x float> %mai.1, ptr addrspace(3) %store.1.addr
  %store.2.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 128
  store <32 x float> %mai.2, ptr addrspace(3) %store.2.addr
  %store.3.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 192
  store <32 x float> %mai.3, ptr addrspace(3) %store.3.addr
  %store.4.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 256
  store <32 x float> %mai.4, ptr addrspace(3) %store.4.addr
  ret void
}

define amdgpu_kernel void @test_iglp_opt_mfma_gemm_attr(ptr addrspace(3) noalias %in, ptr addrspace(3) noalias %out) #1 {
entry:
  call void @llvm.amdgcn.iglp.opt(i32 0)
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %load.0.addr = getelementptr <32 x float>, ptr addrspace(3) %in, i32 %idx
  %load.0 = load <32 x float>, ptr addrspace(3) %load.0.addr
  %load.1.addr = getelementptr <32 x float>, ptr addrspace(3) %load.0.addr, i32 64
  %load.1 = load <32 x float>, ptr addrspace(3) %load.1.addr
  %load.2.addr = getelementptr <32 x float>, ptr addrspace(3) %load.1.addr, i32 128
  %load.2 = load <32 x float>, ptr addrspace(3) %load.2.addr
  %load.3.addr = getelementptr <32 x float>, ptr addrspace(3) %load.2.addr, i32 192
  %load.3 = load <32 x float>, ptr addrspace(3) %load.3.addr
  %load.4.addr = getelementptr <32 x float>, ptr addrspace(3) %load.3.addr, i32 256
  %load.4 = load <32 x float>, ptr addrspace(3) %load.4.addr
  %mai.0 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.0, i32 0, i32 0, i32 0)
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.2, i32 0, i32 0, i32 0)
  %mai.3 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.3, i32 0, i32 0, i32 0)
  %mai.4 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %load.4, i32 0, i32 0, i32 0)
  %store.0.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 %idx
  store <32 x float> %mai.0, ptr addrspace(3) %store.0.addr
  %store.1.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 64
  store <32 x float> %mai.1, ptr addrspace(3) %store.1.addr
  %store.2.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 128
  store <32 x float> %mai.2, ptr addrspace(3) %store.2.addr
  %store.3.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 192
  store <32 x float> %mai.3, ptr addrspace(3) %store.3.addr
  %store.4.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 256
  store <32 x float> %mai.4, ptr addrspace(3) %store.4.addr
  ret void
}

declare void @llvm.amdgcn.iglp.opt(i32) #2
declare i32 @llvm.amdgcn.workitem.id.x() #3
declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32, i32, i32) #3

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,256" }
attributes #1 = { nounwind "amdgpu-flat-work-group-size"="1,256" "amdgpu-igrouplp-small-gemm-rounds"="6" }
attributes #2 = { convergent nounwind willreturn }
attributes #3 = { convergent nounwind readnone }
