; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; Constant folding for llvm.convert.{from,to}.arbitrary.fp scalar calls.
; The folding runs in InstCombinerImpl::visitCallInst; it uses APFloat to
; perform an exact conversion in both directions and handles the saturation
; flag for the TO direction.

declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)
declare half  @llvm.convert.from.arbitrary.fp.f16.i8(i8, metadata)
declare i8    @llvm.convert.to.arbitrary.fp.i8.f32(float, metadata, metadata, i1)
declare i8    @llvm.convert.to.arbitrary.fp.i8.f16(half, metadata, metadata, i1)

; -- Float8E4M3FN (OCP FP8) ----------------------------------------------

; 0x40 = +2.0
; CHECK-LABEL: @from_e4m3_two(
; CHECK-NEXT:    ret float 2.000000e+00
define float @from_e4m3_two() {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 64, metadata !"Float8E4M3FN")
  ret float %r
}

; +2.0 -> 0x40
; CHECK-LABEL: @to_e4m3_two(
; CHECK-NEXT:    ret i8 64
define i8 @to_e4m3_two() {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float 2.0, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

; Saturated overflow: 1e10 -> largest finite E4M3FN = 0x7E = 126 (= 448.0)
; CHECK-LABEL: @to_e4m3_saturate(
; CHECK-NEXT:    ret i8 126
define i8 @to_e4m3_saturate() {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float 1.0e10, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 true)
  ret i8 %r
}

; Non-saturated overflow: 1e10 in a FiniteOnly format produces the NaN
; encoding (0x7F = 127 for E4M3FN).
; CHECK-LABEL: @to_e4m3_nosat(
; CHECK-NEXT:    ret i8 127
define i8 @to_e4m3_nosat() {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float 1.0e10, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

; -- Float8E5M2 (OCP BF8) ------------------------------------------------

; 0x3c = +1.0
; CHECK-LABEL: @from_e5m2_one(
; CHECK-NEXT:    ret float 1.000000e+00
define float @from_e5m2_one() {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 60, metadata !"Float8E5M2")
  ret float %r
}

; 1.25 -> 0x3D = 61 (E5M2 has 2 mantissa bits, so 1.01b is exact).
; CHECK-LABEL: @to_e5m2_1p25(
; CHECK-NEXT:    ret i8 61
define i8 @to_e5m2_1p25() {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float 1.25, metadata !"Float8E5M2", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

; -- Float8E4M3FNUZ (FNUZ family) ----------------------------------------

; FNUZ has a single NaN encoding (sign=1, exp=0, mant=0 = 0x80). A float NaN
; input maps to that encoding.
; CHECK-LABEL: @to_e4m3fnuz_nan(
; CHECK-NEXT:    ret i8 -128
define i8 @to_e4m3fnuz_nan() {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float 0x7FF8000000000000, metadata !"Float8E4M3FNUZ", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

; 0x80 (FNUZ NaN) -> NaN
; CHECK-LABEL: @from_e4m3fnuz_nan(
; CHECK-NEXT:    ret float 0x7FF8000000000000
define float @from_e4m3fnuz_nan() {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 128, metadata !"Float8E4M3FNUZ")
  ret float %r
}

; -- Float8E8M0FNU (exponent-only) ---------------------------------------

; 128 = 2^(128-127) = 2.0
; CHECK-LABEL: @from_e8m0_two(
; CHECK-NEXT:    ret float 2.000000e+00
define float @from_e8m0_two() {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 128, metadata !"Float8E8M0FNU")
  ret float %r
}

; 4.0 = 2^2 -> encoding 129 (0x81, signed as -127)
; CHECK-LABEL: @to_e8m0_four(
; CHECK-NEXT:    ret i8 -127
define i8 @to_e8m0_four() {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float 4.0, metadata !"Float8E8M0FNU", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

; -- Negative cases ------------------------------------------------------

; Non-constant input is not folded.
; CHECK-LABEL: @not_folded(
; CHECK-NEXT:    %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(
define i8 @not_folded(float %x) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float %x, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 false)
  ret i8 %r
}
