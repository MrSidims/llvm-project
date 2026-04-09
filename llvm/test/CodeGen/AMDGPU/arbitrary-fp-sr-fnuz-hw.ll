; RUN: llc -global-isel=0 < %s -mtriple=amdgcn -mcpu=gfx942 | FileCheck -check-prefix=GFX942 %s

; llvm.convert.to.arbitrary.fp.sr for the FNUZ family on gfx942: the same
; amdgcn_cvt_sr_{fp8,bf8}_f32 HW instructions interpret their bits as FNUZ
; on gfx942 (CDNA3), so Float8E4M3FNUZ / Float8E5M2FNUZ map to the HW path
; only on that target.

declare i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float, metadata, i32, i1)

; GFX942-LABEL: sr_fp8_fnuz:
; GFX942: v_cvt_sr_fp8_f32 {{.*}}, v0, v1
define i8 @sr_fp8_fnuz(float %x, i32 %seed) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float %x, metadata !"Float8E4M3FNUZ", i32 %seed, i1 false)
  ret i8 %r
}

; GFX942-LABEL: sr_bf8_fnuz:
; GFX942: v_cvt_sr_bf8_f32 {{.*}}, v0, v1
define i8 @sr_bf8_fnuz(float %x, i32 %seed) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float %x, metadata !"Float8E5M2FNUZ", i32 %seed, i1 false)
  ret i8 %r
}
