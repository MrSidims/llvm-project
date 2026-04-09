; RUN: llc -global-isel=0 < %s -mtriple=amdgcn -mcpu=gfx950 | FileCheck -check-prefix=GFX950 %s

; Test the scaled stochastic-rounding fusion combine: pre-legalize DAG
; combine on ISD::CONVERT_TO_ARBITRARY_FP_SR should recognize
;
;   convert_to_arbitrary_fp_sr(fdiv(val, scale), fp8, seed, sat)
;
; and rewrite it to the AMDGCN amdgcn_cvt_scalef32_sr_{fp8,bf8}_f32
; intrinsic whenever the scale is provably exact (an E8M0 dequantize) or
; the fdiv has the arcp fast-math flag.

declare i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float, metadata, i32, i1)
declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)

; -- Positive: E8M0 scale guarantees exactness ---------------------------

; GFX950-LABEL: sr_fp8_e8m0_scale:
; GFX950: v_cvt_scalef32_sr_fp8_f32
; GFX950-NOT: v_div
; GFX950-NOT: v_cvt_sr_fp8_f32
define i8 @sr_fp8_e8m0_scale(float %val, i8 %scale_bits, i32 %seed) {
  %scale = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %scale_bits, metadata !"Float8E8M0FNU")
  %q = fdiv float %val, %scale
  %r = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float %q, metadata !"Float8E4M3FN", i32 %seed, i1 false)
  ret i8 %r
}

; GFX950-LABEL: sr_bf8_e8m0_scale:
; GFX950: v_cvt_scalef32_sr_bf8_f32
; GFX950-NOT: v_div
define i8 @sr_bf8_e8m0_scale(float %val, i8 %scale_bits, i32 %seed) {
  %scale = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %scale_bits, metadata !"Float8E8M0FNU")
  %q = fdiv float %val, %scale
  %r = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float %q, metadata !"Float8E5M2", i32 %seed, i1 false)
  ret i8 %r
}

; -- Positive: arcp fast-math flag permits the user-opt-in fusion --------

; GFX950-LABEL: sr_fp8_arcp:
; GFX950: v_cvt_scalef32_sr_fp8_f32
; GFX950-NOT: v_div
define i8 @sr_fp8_arcp(float %val, float %scale, i32 %seed) {
  %q = fdiv arcp float %val, %scale
  %r = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float %q, metadata !"Float8E4M3FN", i32 %seed, i1 false)
  ret i8 %r
}

; -- Negative: plain fdiv without arcp must NOT fuse (double rounding) ---

; GFX950-LABEL: sr_no_fuse:
; GFX950: v_div_scale_f32
; GFX950: v_cvt_sr_fp8_f32
; GFX950-NOT: v_cvt_scalef32_sr_fp8_f32
define i8 @sr_no_fuse(float %val, float %scale, i32 %seed) {
  %q = fdiv float %val, %scale
  %r = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float %q, metadata !"Float8E4M3FN", i32 %seed, i1 false)
  ret i8 %r
}
