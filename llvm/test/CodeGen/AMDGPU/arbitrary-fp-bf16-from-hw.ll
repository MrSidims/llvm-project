; RUN: llc -global-isel=0 < %s -mtriple=amdgcn -mcpu=gfx950 | FileCheck -check-prefix=GFX950 %s

; Scalar bf16 / f16 destinations of llvm.convert.from.arbitrary.fp should
; route through amdgcn_cvt_scalef32_pk_{bf16,f16}_{fp8,bf8} with scale=1.0
; on targets with FP8ConversionScaleInsts (gfx950+), avoiding the generic
; bit-manipulation expansion. The low lane of the packed result is
; extracted to produce a single scalar.

declare bfloat @llvm.convert.from.arbitrary.fp.bf16.i8(i8, metadata)
declare half   @llvm.convert.from.arbitrary.fp.f16.i8 (i8, metadata)

; -- bf16 destinations ---------------------------------------------------

; GFX950-LABEL: from_fp8_bf16:
; GFX950: v_cvt_scalef32_pk_bf16_fp8 v0, v0, 1.0
; GFX950-NEXT: s_setpc
define bfloat @from_fp8_bf16(i8 %x) {
  %r = call bfloat @llvm.convert.from.arbitrary.fp.bf16.i8(i8 %x, metadata !"Float8E4M3FN")
  ret bfloat %r
}

; GFX950-LABEL: from_bf8_bf16:
; GFX950: v_cvt_scalef32_pk_bf16_bf8 v0, v0, 1.0
; GFX950-NEXT: s_setpc
define bfloat @from_bf8_bf16(i8 %x) {
  %r = call bfloat @llvm.convert.from.arbitrary.fp.bf16.i8(i8 %x, metadata !"Float8E5M2")
  ret bfloat %r
}

; -- f16 destinations on gfx950 (previously fell back unless gfx1250) ---

; GFX950-LABEL: from_fp8_f16:
; GFX950: v_cvt_scalef32_pk_f16_fp8 v0, v0, 1.0
; GFX950-NEXT: s_setpc
define half @from_fp8_f16(i8 %x) {
  %r = call half @llvm.convert.from.arbitrary.fp.f16.i8(i8 %x, metadata !"Float8E4M3FN")
  ret half %r
}

; GFX950-LABEL: from_bf8_f16:
; GFX950: v_cvt_scalef32_pk_f16_bf8 v0, v0, 1.0
; GFX950-NEXT: s_setpc
define half @from_bf8_f16(i8 %x) {
  %r = call half @llvm.convert.from.arbitrary.fp.f16.i8(i8 %x, metadata !"Float8E5M2")
  ret half %r
}
