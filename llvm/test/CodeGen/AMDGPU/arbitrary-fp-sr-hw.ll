; RUN: llc -global-isel=0 < %s -mtriple=amdgcn -mcpu=gfx950 | FileCheck -check-prefix=GFX950 %s

; Test llvm.convert.to.arbitrary.fp.sr lowering on AMDGPU gfx950. The
; intrinsic lowers to amdgcn_cvt_sr_{fp8,bf8}_f32 for scalar FP8 and to
; amdgcn_cvt_scalef32_sr_pk32_{fp6,bf6}_{f32,f16,bf16} for vector FP6/BF6.
;
; The FNUZ-on-gfx942 path is covered separately because gfx942 reinterprets
; the same HW bits as FNUZ, and mixing both into the same test would cause
; the rejected format to fall back to generic expansion (which is an
; explicit error for the SR node today).

declare i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float, metadata, i32, i1)
declare <6 x i32> @llvm.convert.to.arbitrary.fp.sr.v6i32.v32f32(<32 x float>, metadata, i32, i1)
declare <6 x i32> @llvm.convert.to.arbitrary.fp.sr.v6i32.v32f16(<32 x half>, metadata, i32, i1)
declare <6 x i32> @llvm.convert.to.arbitrary.fp.sr.v6i32.v32bf16(<32 x bfloat>, metadata, i32, i1)

; -- Scalar FP8 OCP ------------------------------------------------------

; GFX950-LABEL: sr_fp8:
; GFX950: v_cvt_sr_fp8_f32 {{.*}}, v0, v1
define i8 @sr_fp8(float %x, i32 %seed) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float %x, metadata !"Float8E4M3FN", i32 %seed, i1 false)
  ret i8 %r
}

; GFX950-LABEL: sr_bf8:
; GFX950: v_cvt_sr_bf8_f32 {{.*}}, v0, v1
define i8 @sr_bf8(float %x, i32 %seed) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float %x, metadata !"Float8E5M2", i32 %seed, i1 false)
  ret i8 %r
}

; -- Vector block-of-32 FP6/BF6 -----------------------------------------

; GFX950-LABEL: sr_fp6_v32f32:
; GFX950: v_cvt_scalef32_sr_pk32_fp6_f32 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+|s[0-9]+}}, 1.0
define <6 x i32> @sr_fp6_v32f32(<32 x float> %x, i32 %seed) {
  %r = call <6 x i32> @llvm.convert.to.arbitrary.fp.sr.v6i32.v32f32(<32 x float> %x, metadata !"Float6E2M3FN", i32 %seed, i1 false)
  ret <6 x i32> %r
}

; GFX950-LABEL: sr_bf6_v32f32:
; GFX950: v_cvt_scalef32_sr_pk32_bf6_f32 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+|s[0-9]+}}, 1.0
define <6 x i32> @sr_bf6_v32f32(<32 x float> %x, i32 %seed) {
  %r = call <6 x i32> @llvm.convert.to.arbitrary.fp.sr.v6i32.v32f32(<32 x float> %x, metadata !"Float6E3M2FN", i32 %seed, i1 false)
  ret <6 x i32> %r
}

; GFX950-LABEL: sr_fp6_v32f16:
; GFX950: v_cvt_scalef32_sr_pk32_fp6_f16 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+|s[0-9]+}}, 1.0
define <6 x i32> @sr_fp6_v32f16(<32 x half> %x, i32 %seed) {
  %r = call <6 x i32> @llvm.convert.to.arbitrary.fp.sr.v6i32.v32f16(<32 x half> %x, metadata !"Float6E2M3FN", i32 %seed, i1 false)
  ret <6 x i32> %r
}

; GFX950-LABEL: sr_bf6_v32bf16:
; GFX950: v_cvt_scalef32_sr_pk32_bf6_bf16 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+|s[0-9]+}}, 1.0
define <6 x i32> @sr_bf6_v32bf16(<32 x bfloat> %x, i32 %seed) {
  %r = call <6 x i32> @llvm.convert.to.arbitrary.fp.sr.v6i32.v32bf16(<32 x bfloat> %x, metadata !"Float6E3M2FN", i32 %seed, i1 false)
  ret <6 x i32> %r
}

; -- Determinism sanity: two calls with the same (value, seed) produce the
; same HW dispatch (CSE folds the duplicate since the intrinsic is pure).

; GFX950-LABEL: sr_determinism:
; GFX950: v_cvt_sr_fp8_f32
; GFX950-NOT: v_cvt_sr_fp8_f32
define { i8, i8 } @sr_determinism(float %x, i32 %seed) {
  %a = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float %x, metadata !"Float8E4M3FN", i32 %seed, i1 false)
  %b = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float %x, metadata !"Float8E4M3FN", i32 %seed, i1 false)
  %s1 = insertvalue { i8, i8 } poison, i8 %a, 0
  %s2 = insertvalue { i8, i8 } %s1, i8 %b, 1
  ret { i8, i8 } %s2
}
