; RUN: opt -S -mtriple=amdgpu7.01-- -loop-reduce < %s | FileCheck -check-prefix=OPT %s
; RUN: opt -S -mtriple=amdgpu9.0a-unknown-amdhsa -loop-reduce < %s | FileCheck -check-prefix=GFX9 %s

; Test that loops with different maximum offsets for different address
; spaces are correctly handled.

; OPT-LABEL: @test_global_addressing_loop_uniform_index_max_offset_i32(
; OPT: {{^}}.lr.ph:
; OPT: %lsr.iv2 = phi ptr addrspace(1) [ %scevgep3, %.lr.ph ], [ %arg1, %.lr.ph.preheader ]
; OPT: %scevgep4 = getelementptr i8, ptr addrspace(1) %lsr.iv2, i64 4095
; OPT: load i8, ptr addrspace(1) %scevgep4, align 1
; OPT: %scevgep3 = getelementptr i8, ptr addrspace(1) %lsr.iv2, i64 1
define amdgpu_kernel void @test_global_addressing_loop_uniform_index_max_offset_i32(ptr addrspace(1) noalias nocapture %arg0, ptr addrspace(1) noalias nocapture readonly %arg1, i32 %n) #0 {
bb:
  %tmp = icmp sgt i32 %n, 0
  br i1 %tmp, label %.lr.ph.preheader, label %._crit_edge

.lr.ph.preheader:                                 ; preds = %bb
  br label %.lr.ph

._crit_edge.loopexit:                             ; preds = %.lr.ph
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %bb
  ret void

.lr.ph:                                           ; preds = %.lr.ph, %.lr.ph.preheader
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %.lr.ph.preheader ]
  %tmp1 = add nuw nsw i64 %indvars.iv, 4095
  %tmp2 = getelementptr inbounds i8, ptr addrspace(1) %arg1, i64 %tmp1
  %tmp3 = load i8, ptr addrspace(1) %tmp2, align 1
  %tmp4 = sext i8 %tmp3 to i32
  %tmp5 = getelementptr inbounds i32, ptr addrspace(1) %arg0, i64 %indvars.iv
  %tmp6 = load i32, ptr addrspace(1) %tmp5, align 4
  %tmp7 = add nsw i32 %tmp6, %tmp4
  store i32 %tmp7, ptr addrspace(1) %tmp5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge.loopexit, label %.lr.ph
}

; OPT-LABEL: @test_global_addressing_loop_uniform_index_max_offset_p1_i32(
; OPT: {{^}}.lr.ph.preheader:
; OPT: %scevgep2 = getelementptr i8, ptr addrspace(1) %arg1, i64 4096
; OPT: br label %.lr.ph

; OPT: {{^}}.lr.ph:
; OPT: %lsr.iv3 = phi ptr addrspace(1) [ %scevgep4, %.lr.ph ], [ %scevgep2, %.lr.ph.preheader ]
; OPT: %scevgep4 = getelementptr i8, ptr addrspace(1) %lsr.iv3, i64 1
define amdgpu_kernel void @test_global_addressing_loop_uniform_index_max_offset_p1_i32(ptr addrspace(1) noalias nocapture %arg0, ptr addrspace(1) noalias nocapture readonly %arg1, i32 %n) #0 {
bb:
  %tmp = icmp sgt i32 %n, 0
  br i1 %tmp, label %.lr.ph.preheader, label %._crit_edge

.lr.ph.preheader:                                 ; preds = %bb
  br label %.lr.ph

._crit_edge.loopexit:                             ; preds = %.lr.ph
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %bb
  ret void

.lr.ph:                                           ; preds = %.lr.ph, %.lr.ph.preheader
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %.lr.ph.preheader ]
  %tmp1 = add nuw nsw i64 %indvars.iv, 4096
  %tmp2 = getelementptr inbounds i8, ptr addrspace(1) %arg1, i64 %tmp1
  %tmp3 = load i8, ptr addrspace(1) %tmp2, align 1
  %tmp4 = sext i8 %tmp3 to i32
  %tmp5 = getelementptr inbounds i32, ptr addrspace(1) %arg0, i64 %indvars.iv
  %tmp6 = load i32, ptr addrspace(1) %tmp5, align 4
  %tmp7 = add nsw i32 %tmp6, %tmp4
  store i32 %tmp7, ptr addrspace(1) %tmp5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge.loopexit, label %.lr.ph
}

; OPT-LABEL: @test_local_addressing_loop_uniform_index_max_offset_i32(
; OPT: {{^}}.lr.ph:
; OPT: %lsr.iv2 = phi ptr addrspace(3) [ %scevgep3, %.lr.ph ], [ %arg1, %.lr.ph.preheader ]
; OPT: %scevgep4 = getelementptr i8, ptr addrspace(3) %lsr.iv2, i32 65535
; OPT: %tmp4 = load i8, ptr addrspace(3) %scevgep4, align 1
; OPT: %scevgep3 = getelementptr i8, ptr addrspace(3) %lsr.iv2, i32 1
define amdgpu_kernel void @test_local_addressing_loop_uniform_index_max_offset_i32(ptr addrspace(1) noalias nocapture %arg0, ptr addrspace(3) noalias nocapture readonly %arg1, i32 %n) #0 {
bb:
  %tmp = icmp sgt i32 %n, 0
  br i1 %tmp, label %.lr.ph.preheader, label %._crit_edge

.lr.ph.preheader:                                 ; preds = %bb
  br label %.lr.ph

._crit_edge.loopexit:                             ; preds = %.lr.ph
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %bb
  ret void

.lr.ph:                                           ; preds = %.lr.ph, %.lr.ph.preheader
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %.lr.ph.preheader ]
  %tmp1 = add nuw nsw i64 %indvars.iv, 65535
  %tmp2 = trunc i64 %tmp1 to i32
  %tmp3 = getelementptr inbounds i8, ptr addrspace(3) %arg1, i32 %tmp2
  %tmp4 = load i8, ptr addrspace(3) %tmp3, align 1
  %tmp5 = sext i8 %tmp4 to i32
  %tmp6 = getelementptr inbounds i32, ptr addrspace(1) %arg0, i64 %indvars.iv
  %tmp7 = load i32, ptr addrspace(1) %tmp6, align 4
  %tmp8 = add nsw i32 %tmp7, %tmp5
  store i32 %tmp8, ptr addrspace(1) %tmp6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge.loopexit, label %.lr.ph
}

; OPT-LABEL: @test_local_addressing_loop_uniform_index_max_offset_p1_i32(
; OPT: {{^}}.lr.ph.preheader:
; OPT: %scevgep2 = getelementptr i8, ptr addrspace(3) %arg1, i32 65536
; OPT: br label %.lr.ph

; OPT: {{^}}.lr.ph:
; OPT: %lsr.iv3 = phi ptr addrspace(3) [ %scevgep4, %.lr.ph ], [ %scevgep2, %.lr.ph.preheader ]
; OPT: %scevgep4 = getelementptr i8, ptr addrspace(3) %lsr.iv3, i32 1
define amdgpu_kernel void @test_local_addressing_loop_uniform_index_max_offset_p1_i32(ptr addrspace(1) noalias nocapture %arg0, ptr addrspace(3) noalias nocapture readonly %arg1, i32 %n) #0 {
bb:
  %tmp = icmp sgt i32 %n, 0
  br i1 %tmp, label %.lr.ph.preheader, label %._crit_edge

.lr.ph.preheader:                                 ; preds = %bb
  br label %.lr.ph

._crit_edge.loopexit:                             ; preds = %.lr.ph
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %bb
  ret void

.lr.ph:                                           ; preds = %.lr.ph, %.lr.ph.preheader
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %.lr.ph.preheader ]
  %tmp1 = add nuw nsw i64 %indvars.iv, 65536
  %tmp2 = trunc i64 %tmp1 to i32
  %tmp3 = getelementptr inbounds i8, ptr addrspace(3) %arg1, i32 %tmp2
  %tmp4 = load i8, ptr addrspace(3) %tmp3, align 1
  %tmp5 = sext i8 %tmp4 to i32
  %tmp6 = getelementptr inbounds i32, ptr addrspace(1) %arg0, i64 %indvars.iv
  %tmp7 = load i32, ptr addrspace(1) %tmp6, align 4
  %tmp8 = add nsw i32 %tmp7, %tmp5
  store i32 %tmp8, ptr addrspace(1) %tmp6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge.loopexit, label %.lr.ph
}

; With three bases and a variable stride the rewrite into pointer induction
; variables is not cheaper than the original addressing and is not applied.

; GFX9-LABEL: @test_global_addressing_loop_three_bases_variable_stride(
; GFX9: {{^}}loop:
; GFX9: %idx = sext i32 %i to i64
; GFX9-NEXT: %acc.p = getelementptr inbounds [16 x i8], ptr addrspace(1) %acc, i64 %idx
; GFX9: %vel.p = getelementptr inbounds [8 x i8], ptr addrspace(1) %vel, i64 %idx
; GFX9: %pos.p = getelementptr inbounds [16 x i8], ptr addrspace(1) %pos, i64 %idx
; GFX9-NOT: lsr.iv
define amdgpu_kernel void @test_global_addressing_loop_three_bases_variable_stride(i32 %n, ptr addrspace(1) %acc, ptr addrspace(1) %vel, ptr addrspace(1) %pos, i32 %start, i32 %stride) #0 {
entry:
  %cmp = icmp slt i32 %start, %n
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ %i.next, %loop ], [ %start, %entry ]
  %idx = sext i32 %i to i64
  %acc.p = getelementptr inbounds [16 x i8], ptr addrspace(1) %acc, i64 %idx
  %acc0 = load float, ptr addrspace(1) %acc.p, align 16
  %acc.p1 = getelementptr inbounds nuw i8, ptr addrspace(1) %acc.p, i64 4
  %acc1 = load float, ptr addrspace(1) %acc.p1, align 4
  %acc.p2 = getelementptr inbounds nuw i8, ptr addrspace(1) %acc.p, i64 8
  %acc2 = load float, ptr addrspace(1) %acc.p2, align 8
  %acc.p3 = getelementptr inbounds nuw i8, ptr addrspace(1) %acc.p, i64 12
  %acc3 = load float, ptr addrspace(1) %acc.p3, align 4
  %vel.p = getelementptr inbounds [8 x i8], ptr addrspace(1) %vel, i64 %idx
  %vel0 = load float, ptr addrspace(1) %vel.p, align 8
  %vel.p1 = getelementptr inbounds nuw i8, ptr addrspace(1) %vel.p, i64 4
  %vel1 = load float, ptr addrspace(1) %vel.p1, align 4
  %pos.p = getelementptr inbounds [16 x i8], ptr addrspace(1) %pos, i64 %idx
  %pos0 = load float, ptr addrspace(1) %pos.p, align 16
  %pos.p1 = getelementptr inbounds nuw i8, ptr addrspace(1) %pos.p, i64 4
  %pos1 = load float, ptr addrspace(1) %pos.p1, align 4
  %pos.p2 = getelementptr inbounds nuw i8, ptr addrspace(1) %pos.p, i64 8
  %pos2 = load float, ptr addrspace(1) %pos.p2, align 8
  %v0 = fadd float %acc0, %vel0
  %v1 = fadd float %acc1, %vel1
  %v2 = fadd float %acc2, %acc3
  %p0 = fadd float %pos0, %v0
  %p1 = fadd float %pos1, %v1
  %p2 = fadd float %pos2, %v2
  store float %p0, ptr addrspace(1) %pos.p, align 16
  store float %p1, ptr addrspace(1) %pos.p1, align 4
  store float %p2, ptr addrspace(1) %pos.p2, align 8
  store float %v0, ptr addrspace(1) %vel.p, align 8
  store float %v1, ptr addrspace(1) %vel.p1, align 4
  store float %v2, ptr addrspace(1) %acc.p3, align 4
  %i.next = add nsw i32 %i, %stride
  %cmp2 = icmp slt i32 %i.next, %n
  br i1 %cmp2, label %loop, label %exit

exit:
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "use-soft-float"="false" }
