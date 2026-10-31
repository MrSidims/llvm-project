; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='print<inline-cost>' -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,ON
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='print<inline-cost>' -amdgpu-inline-count-copies=false -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,OFF
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=inline -S < %s | FileCheck %s --check-prefix=INLINE-ON
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=inline -amdgpu-inline-count-copies=false -S < %s | FileCheck %s --check-prefix=INLINE-OFF

; A call site of a callee with control flow is charged for every copy of the
; callee its caller would hold, one for each call site to the callee, times the
; copies unrolling would make of the loops around it.

define void @callee(ptr %p, i32 %x) #0 {
entry:
  %v = load i32, ptr %p
  %c = icmp eq i32 %v, 0
  br i1 %c, label %then, label %exit

then:
  store i32 %x, ptr %p
  br label %exit

exit:
  ret void
}

define void @straight(ptr %p, i32 %x) #0 {
  store i32 %x, ptr %p
  ret void
}

; CHECK-LABEL: Analyzing call of callee... (caller:siblings)
; ON: {{^ *}}Cost: 150{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
; CHECK: Analyzing call of callee... (caller:siblings)
; ON: {{^ *}}Cost: 150{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
; CHECK: Analyzing call of callee... (caller:siblings)
; ON: {{^ *}}Cost: 150{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
define void @siblings(ptr %p, i32 %x) {
  call void @callee(ptr %p, i32 %x)
  call void @callee(ptr %p, i32 %x)
  call void @callee(ptr %p, i32 %x)
  ret void
}

; A call site that is never inlined holds no copy of the callee.
; CHECK-LABEL: Analyzing call of callee... (caller:siblings_noinline)
; ON: {{^ *}}Cost: 100{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
; CHECK: Analyzing call of callee... (caller:siblings_noinline)
; ON: {{^ *}}Cost: 100{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
define void @siblings_noinline(ptr %p, i32 %x) {
  call void @callee(ptr %p, i32 %x)
  call void @callee(ptr %p, i32 %x)
  call void @callee(ptr %p, i32 %x) #2
  ret void
}

; CHECK-LABEL: Analyzing call of callee... (caller:loop_8)
; ON: {{^ *}}Cost: 400{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
define void @loop_8(ptr %p) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @callee(ptr %p, i32 %i)
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; The header of a loop that is not rotated runs once more than its body.
; CHECK-LABEL: Analyzing call of callee... (caller:loop_8_header_exit)
; ON: {{^ *}}Cost: 400{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
define void @loop_8_header_exit(ptr %p) {
entry:
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
  %done = icmp eq i32 %i, 8
  br i1 %done, label %exit, label %body

body:
  call void @callee(ptr %p, i32 %i)
  %i.next = add nuw nsw i32 %i, 1
  br label %header

exit:
  ret void
}

; CHECK-LABEL: Analyzing call of callee... (caller:nested_4_2)
; ON: {{^ *}}Cost: 400{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
define void @nested_4_2(ptr %p) {
entry:
  br label %outer

outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner ]
  call void @callee(ptr %p, i32 %j)
  %j.next = add nuw nsw i32 %j, 1
  %inner.done = icmp eq i32 %j.next, 4
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i32 %i, 1
  %outer.done = icmp eq i32 %i.next, 2
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

; The unrolled inner loop makes the outer loop too large to be unrolled.
; CHECK-LABEL: Analyzing call of callee... (caller:nested_8_8)
; ON: {{^ *}}Cost: 400{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
define void @nested_8_8(ptr %p) {
entry:
  br label %outer

outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner ]
  call void @callee(ptr %p, i32 %j)
  %j.next = add nuw nsw i32 %j, 1
  %inner.done = icmp eq i32 %j.next, 8
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i32 %i, 1
  %outer.done = icmp eq i32 %i.next, 8
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

; CHECK-LABEL: Analyzing call of callee... (caller:loop_n)
; CHECK: {{^ *}}Cost: 50{{$}}
define void @loop_n(ptr %p, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @callee(ptr %p, i32 %i)
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Too large to be fully unrolled.
; CHECK-LABEL: Analyzing call of callee... (caller:loop_100)
; CHECK: {{^ *}}Cost: 50{{$}}
define void @loop_100(ptr %p) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @callee(ptr %p, i32 %i)
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 100
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; A loop over a private array is unrolled with a higher threshold.
; CHECK-LABEL: Analyzing call of callee... (caller:private_100)
; ON: {{^ *}}Cost: 5000{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
define void @private_100(ptr %p) {
entry:
  %a = alloca [100 x i32], align 4, addrspace(5)
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %gep = getelementptr inbounds i32, ptr addrspace(5) %a, i32 %i
  store i32 %i, ptr addrspace(5) %gep
  call void @callee(ptr %p, i32 %i)
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 100
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: Analyzing call of callee... (caller:loop_8_optsize)
; CHECK: {{^ *}}Cost: 50{{$}}
define void @loop_8_optsize(ptr %p) optsize {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @callee(ptr %p, i32 %i)
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: Analyzing call of callee... (caller:loop_8_disable)
; CHECK: {{^ *}}Cost: 50{{$}}
define void @loop_8_disable(ptr %p) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @callee(ptr %p, i32 %i)
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 8
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

; CHECK-LABEL: Analyzing call of callee... (caller:count_4)
; ON: {{^ *}}Cost: 200{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
define void @count_4(ptr %p, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @callee(ptr %p, i32 %i)
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !2

exit:
  ret void
}

; CHECK-LABEL: Analyzing call of callee... (caller:full_100)
; ON: {{^ *}}Cost: 5000{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
define void @full_100(ptr %p) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @callee(ptr %p, i32 %i)
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 100
  br i1 %done, label %exit, label %loop, !llvm.loop !4

exit:
  ret void
}

; The pragma raises the threshold of full unrolling.
; CHECK-LABEL: Analyzing call of callee... (caller:enable_100)
; ON: {{^ *}}Cost: 5000{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
define void @enable_100(ptr %p) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @callee(ptr %p, i32 %i)
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 100
  br i1 %done, label %exit, label %loop, !llvm.loop !6

exit:
  ret void
}

; Too large to be fully unrolled even with the raised threshold.
; CHECK-LABEL: Analyzing call of callee... (caller:enable_2000)
; CHECK: {{^ *}}Cost: 50{{$}}
define void @enable_2000(ptr %p) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @callee(ptr %p, i32 %i)
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 2000
  br i1 %done, label %exit, label %loop, !llvm.loop !8

exit:
  ret void
}

; A callee without control flow is not charged.
; CHECK-LABEL: Analyzing call of straight... (caller:loop_8_straight)
; CHECK: {{^ *}}Cost: 50{{$}}
define void @loop_8_straight(ptr %p) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @straight(ptr %p, i32 %i)
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; The copies are counted up to the threshold.
; CHECK-LABEL: Analyzing call of callee... (caller:limit)
; ON: {{^ *}}Cost: 1000{{$}}
; OFF: {{^ *}}Cost: 50{{$}}
; CHECK: {{^ *}}Threshold: 1000{{$}}

; INLINE-ON-LABEL: define void @limit(
; INLINE-ON: call void @callee(

; INLINE-OFF-LABEL: define void @limit(
; INLINE-OFF-NOT: call void @callee(
; INLINE-OFF: ret void
define void @limit(ptr %p) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @callee(ptr %p, i32 %i) #1
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 100
  br i1 %done, label %exit, label %loop, !llvm.loop !9

exit:
  ret void
}

attributes #0 = { "function-inline-cost"="50" "function-inline-threshold"="100000" }
attributes #1 = { "function-inline-threshold"="1000" }
attributes #2 = { noinline }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.unroll.count", i32 4}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.unroll.full"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.unroll.enable"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !5}
