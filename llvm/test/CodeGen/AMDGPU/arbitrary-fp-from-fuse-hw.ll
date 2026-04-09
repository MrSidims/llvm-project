; RUN: llc -global-isel=0 < %s -mtriple=amdgcn -mcpu=gfx950 | FileCheck -check-prefix=GFX950 %s

; Test the FROM+fmul fusion combine: matchFMulArbitraryFPScaleFrom should
; recognize fmul(convert_from_arbitrary_fp(bits, fp8), scale) and fuse
; into the AMDGCN amdgcn_cvt_scalef32_{f32,pk_f32}_{fp8,bf8} intrinsics.
;
; This covers the generic ISD::CONVERT_FROM_ARBITRARY_FP form, which is
; what frontends emitting the new llvm.convert.from.arbitrary.fp intrinsic
; produce. The AMDGCN-intrinsic form (emitted by ROCDL translators) is
; covered by the sibling test scaled-arbitrary-fp-fp8-hw.ll.
;
; Fusion is always safe in the FROM direction: fp8→f32 widening is exact,
; so running the fmul as part of the HW op does the same amount of
; rounding as a separate fmul would.

declare float       @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)
declare <2 x float> @llvm.convert.from.arbitrary.fp.v2f32.i16(i16, metadata)

; -- Scalar f32 --------------------------------------------------------

; GFX950-LABEL: from_fp8_scalar_fuse:
; GFX950: v_cvt_scalef32_f32_fp8
; GFX950-NOT: v_cvt_f32_fp8
; GFX950-NOT: v_mul_f32
define float @from_fp8_scalar_fuse(i8 %b, float %scale) {
  %f = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %b, metadata !"Float8E4M3FN")
  %r = fmul float %f, %scale
  ret float %r
}

; GFX950-LABEL: from_bf8_scalar_fuse:
; GFX950: v_cvt_scalef32_f32_bf8
; GFX950-NOT: v_cvt_f32_bf8
define float @from_bf8_scalar_fuse(i8 %b, float %scale) {
  %f = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %b, metadata !"Float8E5M2")
  %r = fmul float %f, %scale
  ret float %r
}

; -- Packed v2f32 with splat scale ------------------------------------

; GFX950-LABEL: from_fp8_v2_splat:
; GFX950: v_cvt_scalef32_pk_f32_fp8
; GFX950-NOT: v_cvt_f32_fp8
; GFX950-NOT: v_pk_mul_f32
define <2 x float> @from_fp8_v2_splat(i16 %packed, float %scale) {
  %f = call <2 x float> @llvm.convert.from.arbitrary.fp.v2f32.i16(i16 %packed, metadata !"Float8E4M3FN")
  %s0 = insertelement <2 x float> poison, float %scale, i32 0
  %s = shufflevector <2 x float> %s0, <2 x float> poison, <2 x i32> zeroinitializer
  %r = fmul <2 x float> %f, %s
  ret <2 x float> %r
}

; GFX950-LABEL: from_bf8_v2_splat:
; GFX950: v_cvt_scalef32_pk_f32_bf8
define <2 x float> @from_bf8_v2_splat(i16 %packed, float %scale) {
  %f = call <2 x float> @llvm.convert.from.arbitrary.fp.v2f32.i16(i16 %packed, metadata !"Float8E5M2")
  %s0 = insertelement <2 x float> poison, float %scale, i32 0
  %s = shufflevector <2 x float> %s0, <2 x float> poison, <2 x i32> zeroinitializer
  %r = fmul <2 x float> %f, %s
  ret <2 x float> %r
}

; -- Packed v2f32 with E8M0-dequantized scale (natural MX shape) -------

; GFX950-LABEL: from_fp8_v2_e8m0:
; GFX950: v_cvt_scalef32_pk_f32_fp8
; GFX950-NOT: v_cvt_f32_fp8
define <2 x float> @from_fp8_v2_e8m0(i16 %packed, i8 %scale_bits) {
  %f = call <2 x float> @llvm.convert.from.arbitrary.fp.v2f32.i16(i16 %packed, metadata !"Float8E4M3FN")
  %scale = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %scale_bits, metadata !"Float8E8M0FNU")
  %s0 = insertelement <2 x float> poison, float %scale, i32 0
  %s = shufflevector <2 x float> %s0, <2 x float> poison, <2 x i32> zeroinitializer
  %r = fmul <2 x float> %f, %s
  ret <2 x float> %r
}

; -- Negative: non-splat v2 scale must NOT fuse into packed form ------

; GFX950-LABEL: from_fp8_v2_nonsplat:
; GFX950-NOT: v_cvt_scalef32_pk_f32_fp8
; GFX950: v_pk_mul_f32
define <2 x float> @from_fp8_v2_nonsplat(i16 %packed, <2 x float> %scale) {
  %f = call <2 x float> @llvm.convert.from.arbitrary.fp.v2f32.i16(i16 %packed, metadata !"Float8E4M3FN")
  %r = fmul <2 x float> %f, %scale
  ret <2 x float> %r
}
