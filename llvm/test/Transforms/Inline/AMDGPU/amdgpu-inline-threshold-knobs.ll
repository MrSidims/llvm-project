; REQUIRES: asserts
; RUN: opt -mtriple=amdgpu-amd-amdhsa -S -passes=inline -inline-threshold=100 -debug-only=inline < %s 2>&1 | FileCheck -check-prefixes=CHECK,DEFAULT %s

; The knobs are read from the callee, so an explicitly given flag must also
; override an attribute placed on a callee.
; RUN: opt -mtriple=amdgpu-amd-amdhsa -S -passes=inline -inline-threshold=100 -debug-only=inline \
; RUN:   -amdgpu-inline-threshold-multiplier=11 -amdgpu-inline-sgprs-until-spill=26 \
; RUN:   -amdgpu-inline-vgprs-until-spill=32 < %s 2>&1 | FileCheck -check-prefixes=CHECK,EXPLICIT %s

; RUN: opt -mtriple=amdgpu-amd-amdhsa -S -passes=inline -inline-threshold=100 -debug-only=inline \
; RUN:   -amdgpu-inline-threshold-multiplier=3 -amdgpu-inline-sgprs-until-spill=4 \
; RUN:   -amdgpu-inline-vgprs-until-spill=34 < %s 2>&1 | FileCheck -check-prefix=FLAG %s

; CHECK:    threshold=1650), Call:   %s = call noundef i32 @sgpr_args(
; CHECK:    threshold=2145), Call:   %v = call noundef i32 @vgpr_args(
; DEFAULT:  threshold=480), Call:   %sa = call noundef i32 @sgpr_args_attrs(
; DEFAULT:  threshold=1650), Call:   %va = call noundef i32 @vgpr_args_attrs(
; EXPLICIT: threshold=1650), Call:   %sa = call noundef i32 @sgpr_args_attrs(
; EXPLICIT: threshold=2145), Call:   %va = call noundef i32 @vgpr_args_attrs(

; FLAG:     threshold=720), Call:   %s = call noundef i32 @sgpr_args(
; FLAG:     threshold=450), Call:   %v = call noundef i32 @vgpr_args(
; FLAG:     threshold=720), Call:   %sa = call noundef i32 @sgpr_args_attrs(
; FLAG:     threshold=450), Call:   %va = call noundef i32 @vgpr_args_attrs(

define noundef i32 @sgpr_args(i32 inreg noundef %a0, i32 inreg noundef %a1, i32 inreg noundef %a2, i32 inreg noundef %a3, i32 inreg noundef %a4, i32 inreg noundef %a5, i32 inreg noundef %a6, i32 inreg noundef %a7) {
  %x = xor i32 %a0, %a1
  ret i32 %x
}

define noundef i32 @vgpr_args(i32 noundef %a0, i32 noundef %a1, i32 noundef %a2, i32 noundef %a3, i32 noundef %a4, i32 noundef %a5, i32 noundef %a6, i32 noundef %a7, i32 noundef %a8, i32 noundef %a9, i32 noundef %a10, i32 noundef %a11, i32 noundef %a12, i32 noundef %a13, i32 noundef %a14, i32 noundef %a15, i32 noundef %a16, i32 noundef %a17, i32 noundef %a18, i32 noundef %a19, i32 noundef %a20, i32 noundef %a21, i32 noundef %a22, i32 noundef %a23, i32 noundef %a24, i32 noundef %a25, i32 noundef %a26, i32 noundef %a27, i32 noundef %a28, i32 noundef %a29, i32 noundef %a30, i32 noundef %a31, i32 noundef %a32, i32 noundef %a33) {
  %x = xor i32 %a0, %a1
  ret i32 %x
}

define noundef i32 @sgpr_args_attrs(i32 inreg noundef %a0, i32 inreg noundef %a1, i32 inreg noundef %a2, i32 inreg noundef %a3, i32 inreg noundef %a4, i32 inreg noundef %a5, i32 inreg noundef %a6, i32 inreg noundef %a7) #0 {
  %x = xor i32 %a0, %a1
  ret i32 %x
}

define noundef i32 @vgpr_args_attrs(i32 noundef %a0, i32 noundef %a1, i32 noundef %a2, i32 noundef %a3, i32 noundef %a4, i32 noundef %a5, i32 noundef %a6, i32 noundef %a7, i32 noundef %a8, i32 noundef %a9, i32 noundef %a10, i32 noundef %a11, i32 noundef %a12, i32 noundef %a13, i32 noundef %a14, i32 noundef %a15, i32 noundef %a16, i32 noundef %a17, i32 noundef %a18, i32 noundef %a19, i32 noundef %a20, i32 noundef %a21, i32 noundef %a22, i32 noundef %a23, i32 noundef %a24, i32 noundef %a25, i32 noundef %a26, i32 noundef %a27, i32 noundef %a28, i32 noundef %a29, i32 noundef %a30, i32 noundef %a31, i32 noundef %a32, i32 noundef %a33) #1 {
  %x = xor i32 %a0, %a1
  ret i32 %x
}

define noundef i32 @caller(i32 noundef %a0, i32 noundef %a1, i32 noundef %a2, i32 noundef %a3, i32 noundef %a4, i32 noundef %a5, i32 noundef %a6, i32 noundef %a7, i32 noundef %a8, i32 noundef %a9, i32 noundef %a10, i32 noundef %a11, i32 noundef %a12, i32 noundef %a13, i32 noundef %a14, i32 noundef %a15, i32 noundef %a16, i32 noundef %a17, i32 noundef %a18, i32 noundef %a19, i32 noundef %a20, i32 noundef %a21, i32 noundef %a22, i32 noundef %a23, i32 noundef %a24, i32 noundef %a25, i32 noundef %a26, i32 noundef %a27, i32 noundef %a28, i32 noundef %a29, i32 noundef %a30, i32 noundef %a31, i32 noundef %a32, i32 noundef %a33) {
  %s = call noundef i32 @sgpr_args(i32 inreg noundef %a0, i32 inreg noundef %a1, i32 inreg noundef %a2, i32 inreg noundef %a3, i32 inreg noundef %a4, i32 inreg noundef %a5, i32 inreg noundef %a6, i32 inreg noundef %a7)
  %v = call noundef i32 @vgpr_args(i32 noundef %a0, i32 noundef %a1, i32 noundef %a2, i32 noundef %a3, i32 noundef %a4, i32 noundef %a5, i32 noundef %a6, i32 noundef %a7, i32 noundef %a8, i32 noundef %a9, i32 noundef %a10, i32 noundef %a11, i32 noundef %a12, i32 noundef %a13, i32 noundef %a14, i32 noundef %a15, i32 noundef %a16, i32 noundef %a17, i32 noundef %a18, i32 noundef %a19, i32 noundef %a20, i32 noundef %a21, i32 noundef %a22, i32 noundef %a23, i32 noundef %a24, i32 noundef %a25, i32 noundef %a26, i32 noundef %a27, i32 noundef %a28, i32 noundef %a29, i32 noundef %a30, i32 noundef %a31, i32 noundef %a32, i32 noundef %a33)
  %sa = call noundef i32 @sgpr_args_attrs(i32 inreg noundef %a0, i32 inreg noundef %a1, i32 inreg noundef %a2, i32 inreg noundef %a3, i32 inreg noundef %a4, i32 inreg noundef %a5, i32 inreg noundef %a6, i32 inreg noundef %a7)
  %va = call noundef i32 @vgpr_args_attrs(i32 noundef %a0, i32 noundef %a1, i32 noundef %a2, i32 noundef %a3, i32 noundef %a4, i32 noundef %a5, i32 noundef %a6, i32 noundef %a7, i32 noundef %a8, i32 noundef %a9, i32 noundef %a10, i32 noundef %a11, i32 noundef %a12, i32 noundef %a13, i32 noundef %a14, i32 noundef %a15, i32 noundef %a16, i32 noundef %a17, i32 noundef %a18, i32 noundef %a19, i32 noundef %a20, i32 noundef %a21, i32 noundef %a22, i32 noundef %a23, i32 noundef %a24, i32 noundef %a25, i32 noundef %a26, i32 noundef %a27, i32 noundef %a28, i32 noundef %a29, i32 noundef %a30, i32 noundef %a31, i32 noundef %a32, i32 noundef %a33)
  %r0 = xor i32 %s, %v
  %r1 = xor i32 %r0, %sa
  %r2 = xor i32 %r1, %va
  ret i32 %r2
}

attributes #0 = { "amdgpu-inline-threshold-multiplier"="2" "amdgpu-inline-sgprs-until-spill"="4" }
attributes #1 = { "amdgpu-inline-vgprs-until-spill"="34" }
