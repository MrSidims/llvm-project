; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='print<inline-cost>' -disable-output < %s 2>&1 | FileCheck %s

; An fneg that folds into a source modifier of its user adds no cost.

; CHECK-LABEL: Analyzing call of fneg_f32... (caller:caller_f32)
; CHECK-NEXT: define float @fneg_f32(float %a, float %b) {
; CHECK-NEXT: cost delta = 0{{$}}
; CHECK-NEXT: %neg = fneg float %a
; CHECK: NumInstructionsSimplified: 2
define float @fneg_f32(float %a, float %b) {
  %neg = fneg float %a
  %mul = fmul float %neg, %b
  ret float %mul
}

define float @caller_f32(float %a, float %b) {
  %r = call float @fneg_f32(float %a, float %b)
  ret float %r
}

; CHECK-LABEL: Analyzing call of fneg_f64... (caller:caller_f64)
; CHECK-NEXT: define double @fneg_f64(double %a, double %b) {
; CHECK-NEXT: cost delta = 0{{$}}
; CHECK-NEXT: %neg = fneg double %a
; CHECK: NumInstructionsSimplified: 2
define double @fneg_f64(double %a, double %b) {
  %neg = fneg double %a
  %add = fadd double %neg, %b
  ret double %add
}

define double @caller_f64(double %a, double %b) {
  %r = call double @fneg_f64(double %a, double %b)
  ret double %r
}
