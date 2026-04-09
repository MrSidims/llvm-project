	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	to_fp8_rne                      ; -- Begin function to_fp8_rne
	.p2align	6
	.type	to_fp8_rne,@function
to_fp8_rne:                             ; @to_fp8_rne
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_mov_b32_e32 v1, 0
	v_cvt_pk_fp8_f32 v1, v0, s0
	v_and_b32_e32 v0, 0xff, v1
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	to_fp8_rne, .Lfunc_end0-to_fp8_rne
                                        ; -- End function
	.set to_fp8_rne.num_vgpr, 2
	.set to_fp8_rne.num_agpr, 0
	.set to_fp8_rne.numbered_sgpr, 32
	.set to_fp8_rne.num_named_barrier, 0
	.set to_fp8_rne.private_seg_size, 0
	.set to_fp8_rne.uses_vcc, 0
	.set to_fp8_rne.uses_flat_scratch, 0
	.set to_fp8_rne.has_dyn_sized_stack, 0
	.set to_fp8_rne.has_recursion, 0
	.set to_fp8_rne.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 28
; TotalNumSgprs: 38
; NumVgprs: 2
; NumAgprs: 0
; TotalNumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
	.text
	.globl	to_bf8_rne                      ; -- Begin function to_bf8_rne
	.p2align	6
	.type	to_bf8_rne,@function
to_bf8_rne:                             ; @to_bf8_rne
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_mov_b32_e32 v1, 0
	v_cvt_pk_bf8_f32 v1, v0, s0
	v_and_b32_e32 v0, 0xff, v1
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	to_bf8_rne, .Lfunc_end1-to_bf8_rne
                                        ; -- End function
	.set to_bf8_rne.num_vgpr, 2
	.set to_bf8_rne.num_agpr, 0
	.set to_bf8_rne.numbered_sgpr, 32
	.set to_bf8_rne.num_named_barrier, 0
	.set to_bf8_rne.private_seg_size, 0
	.set to_bf8_rne.uses_vcc, 0
	.set to_bf8_rne.uses_flat_scratch, 0
	.set to_bf8_rne.has_dyn_sized_stack, 0
	.set to_bf8_rne.has_recursion, 0
	.set to_bf8_rne.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 28
; TotalNumSgprs: 38
; NumVgprs: 2
; NumAgprs: 0
; TotalNumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
	.text
	.globl	to_fp8_rne_sat                  ; -- Begin function to_fp8_rne_sat
	.p2align	6
	.type	to_fp8_rne_sat,@function
to_fp8_rne_sat:                         ; @to_fp8_rne_sat
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_mov_b32_e32 v1, 0
	v_cvt_pk_fp8_f32 v1, v0, s0
	v_and_b32_e32 v0, 0xff, v1
	s_setpc_b64 s[30:31]
.Lfunc_end2:
	.size	to_fp8_rne_sat, .Lfunc_end2-to_fp8_rne_sat
                                        ; -- End function
	.set to_fp8_rne_sat.num_vgpr, 2
	.set to_fp8_rne_sat.num_agpr, 0
	.set to_fp8_rne_sat.numbered_sgpr, 32
	.set to_fp8_rne_sat.num_named_barrier, 0
	.set to_fp8_rne_sat.private_seg_size, 0
	.set to_fp8_rne_sat.uses_vcc, 0
	.set to_fp8_rne_sat.uses_flat_scratch, 0
	.set to_fp8_rne_sat.has_dyn_sized_stack, 0
	.set to_fp8_rne_sat.has_recursion, 0
	.set to_fp8_rne_sat.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 28
; TotalNumSgprs: 38
; NumVgprs: 2
; NumAgprs: 0
; TotalNumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
	.text
	.globl	to_fp8_from_f16_rne             ; -- Begin function to_fp8_from_f16_rne
	.p2align	6
	.type	to_fp8_from_f16_rne,@function
to_fp8_from_f16_rne:                    ; @to_fp8_from_f16_rne
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_cvt_f32_f16_e32 v0, v0
	v_mov_b32_e32 v1, 0
	v_cvt_pk_fp8_f32 v1, v0, s0
	v_and_b32_e32 v0, 0xff, v1
	s_setpc_b64 s[30:31]
.Lfunc_end3:
	.size	to_fp8_from_f16_rne, .Lfunc_end3-to_fp8_from_f16_rne
                                        ; -- End function
	.set to_fp8_from_f16_rne.num_vgpr, 2
	.set to_fp8_from_f16_rne.num_agpr, 0
	.set to_fp8_from_f16_rne.numbered_sgpr, 32
	.set to_fp8_from_f16_rne.num_named_barrier, 0
	.set to_fp8_from_f16_rne.private_seg_size, 0
	.set to_fp8_from_f16_rne.uses_vcc, 0
	.set to_fp8_from_f16_rne.uses_flat_scratch, 0
	.set to_fp8_from_f16_rne.has_dyn_sized_stack, 0
	.set to_fp8_from_f16_rne.has_recursion, 0
	.set to_fp8_from_f16_rne.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 32
; TotalNumSgprs: 38
; NumVgprs: 2
; NumAgprs: 0
; TotalNumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
	.text
	.globl	to_fp8_v2_store                 ; -- Begin function to_fp8_v2_store
	.p2align	6
	.type	to_fp8_v2_store,@function
to_fp8_v2_store:                        ; @to_fp8_v2_store
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_mov_b32_e32 v4, 0
	v_cvt_pk_fp8_f32 v4, v0, v1
	global_store_short v[2:3], v4, off
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end4:
	.size	to_fp8_v2_store, .Lfunc_end4-to_fp8_v2_store
                                        ; -- End function
	.set to_fp8_v2_store.num_vgpr, 5
	.set to_fp8_v2_store.num_agpr, 0
	.set to_fp8_v2_store.numbered_sgpr, 32
	.set to_fp8_v2_store.num_named_barrier, 0
	.set to_fp8_v2_store.private_seg_size, 0
	.set to_fp8_v2_store.uses_vcc, 0
	.set to_fp8_v2_store.uses_flat_scratch, 0
	.set to_fp8_v2_store.has_dyn_sized_stack, 0
	.set to_fp8_v2_store.has_recursion, 0
	.set to_fp8_v2_store.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 32
; TotalNumSgprs: 38
; NumVgprs: 5
; NumAgprs: 0
; TotalNumVgprs: 5
; ScratchSize: 0
; MemoryBound: 0
	.text
	.globl	to_bf8_v2_store                 ; -- Begin function to_bf8_v2_store
	.p2align	6
	.type	to_bf8_v2_store,@function
to_bf8_v2_store:                        ; @to_bf8_v2_store
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_mov_b32_e32 v4, 0
	v_cvt_pk_bf8_f32 v4, v0, v1
	global_store_short v[2:3], v4, off
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end5:
	.size	to_bf8_v2_store, .Lfunc_end5-to_bf8_v2_store
                                        ; -- End function
	.set to_bf8_v2_store.num_vgpr, 5
	.set to_bf8_v2_store.num_agpr, 0
	.set to_bf8_v2_store.numbered_sgpr, 32
	.set to_bf8_v2_store.num_named_barrier, 0
	.set to_bf8_v2_store.private_seg_size, 0
	.set to_bf8_v2_store.uses_vcc, 0
	.set to_bf8_v2_store.uses_flat_scratch, 0
	.set to_bf8_v2_store.has_dyn_sized_stack, 0
	.set to_bf8_v2_store.has_recursion, 0
	.set to_bf8_v2_store.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 32
; TotalNumSgprs: 38
; NumVgprs: 5
; NumAgprs: 0
; TotalNumVgprs: 5
; ScratchSize: 0
; MemoryBound: 0
	.text
	.globl	to_fp4_rne                      ; -- Begin function to_fp4_rne
	.p2align	6
	.type	to_fp4_rne,@function
to_fp4_rne:                             ; @to_fp4_rne
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_mov_b32_e32 v1, 0
	v_cvt_scalef32_pk_fp4_f32 v1, v0, s0, 1.0
	v_and_b32_e32 v0, 15, v1
	s_setpc_b64 s[30:31]
.Lfunc_end6:
	.size	to_fp4_rne, .Lfunc_end6-to_fp4_rne
                                        ; -- End function
	.set to_fp4_rne.num_vgpr, 2
	.set to_fp4_rne.num_agpr, 0
	.set to_fp4_rne.numbered_sgpr, 32
	.set to_fp4_rne.num_named_barrier, 0
	.set to_fp4_rne.private_seg_size, 0
	.set to_fp4_rne.uses_vcc, 0
	.set to_fp4_rne.uses_flat_scratch, 0
	.set to_fp4_rne.has_dyn_sized_stack, 0
	.set to_fp4_rne.has_recursion, 0
	.set to_fp4_rne.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 24
; TotalNumSgprs: 38
; NumVgprs: 2
; NumAgprs: 0
; TotalNumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
	.text
	.globl	to_fp8_v2_from_f16_store        ; -- Begin function to_fp8_v2_from_f16_store
	.p2align	6
	.type	to_fp8_v2_from_f16_store,@function
to_fp8_v2_from_f16_store:               ; @to_fp8_v2_from_f16_store
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v2, v1
	v_cvt_f32_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v0, v0
	v_mov_b32_e32 v4, 0
	v_cvt_pk_fp8_f32 v4, v0, v1
	global_store_short v[2:3], v4, off
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end7:
	.size	to_fp8_v2_from_f16_store, .Lfunc_end7-to_fp8_v2_from_f16_store
                                        ; -- End function
	.set to_fp8_v2_from_f16_store.num_vgpr, 5
	.set to_fp8_v2_from_f16_store.num_agpr, 0
	.set to_fp8_v2_from_f16_store.numbered_sgpr, 32
	.set to_fp8_v2_from_f16_store.num_named_barrier, 0
	.set to_fp8_v2_from_f16_store.private_seg_size, 0
	.set to_fp8_v2_from_f16_store.uses_vcc, 0
	.set to_fp8_v2_from_f16_store.uses_flat_scratch, 0
	.set to_fp8_v2_from_f16_store.has_dyn_sized_stack, 0
	.set to_fp8_v2_from_f16_store.has_recursion, 0
	.set to_fp8_v2_from_f16_store.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 52
; TotalNumSgprs: 38
; NumVgprs: 5
; NumAgprs: 0
; TotalNumVgprs: 5
; ScratchSize: 0
; MemoryBound: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 5
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
