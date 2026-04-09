; RUN: llc -global-isel=0 < %s -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 | FileCheck %s --check-prefix=GFX950

; Test that the FP6 / BF6 vector paths of llvm.convert.{from,to}.arbitrary.fp
; lower to the gfx950+ pk32 (and 2xpk16) block-of-32 conversion instructions
; for the natural MX-block size.
;
; FROM direction: <6 x i32> -> <32 x {f32,f16,bf16}>
;   v_cvt_scalef32_pk32_f32_{fp6,bf6}
;   v_cvt_scalef32_pk32_f16_{fp6,bf6}
;   v_cvt_scalef32_pk32_bf16_{fp6,bf6}
;
; TO direction: <32 x {f32,f16,bf16}> -> <6 x i32>
;   v_cvt_scalef32_2xpk16_{fp6,bf6}_f32 (f32 input; pk32 form is SR-only)
;   v_cvt_scalef32_pk32_{fp6,bf6}_f16
;   v_cvt_scalef32_pk32_{fp6,bf6}_bf16

declare <32 x float>  @llvm.convert.from.arbitrary.fp.v32f32.v6i32(<6 x i32>, metadata)
declare <32 x half>   @llvm.convert.from.arbitrary.fp.v32f16.v6i32(<6 x i32>, metadata)
declare <32 x bfloat> @llvm.convert.from.arbitrary.fp.v32bf16.v6i32(<6 x i32>, metadata)

declare <6 x i32> @llvm.convert.to.arbitrary.fp.v6i32.v32f32 (<32 x float>,  metadata, metadata, i1)
declare <6 x i32> @llvm.convert.to.arbitrary.fp.v6i32.v32f16 (<32 x half>,   metadata, metadata, i1)
declare <6 x i32> @llvm.convert.to.arbitrary.fp.v6i32.v32bf16(<32 x bfloat>, metadata, metadata, i1)

; GFX950-LABEL: from_fp6_f32:
; GFX950: v_cvt_scalef32_pk32_f32_fp6 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 1.0
define <32 x float> @from_fp6_f32(<6 x i32> %x) {
  %r = call <32 x float> @llvm.convert.from.arbitrary.fp.v32f32.v6i32(<6 x i32> %x, metadata !"Float6E2M3FN")
  ret <32 x float> %r
}

; GFX950-LABEL: from_bf6_f32:
; GFX950: v_cvt_scalef32_pk32_f32_bf6 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 1.0
define <32 x float> @from_bf6_f32(<6 x i32> %x) {
  %r = call <32 x float> @llvm.convert.from.arbitrary.fp.v32f32.v6i32(<6 x i32> %x, metadata !"Float6E3M2FN")
  ret <32 x float> %r
}

; GFX950-LABEL: from_fp6_f16:
; GFX950: v_cvt_scalef32_pk32_f16_fp6 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 1.0
define <32 x half> @from_fp6_f16(<6 x i32> %x) {
  %r = call <32 x half> @llvm.convert.from.arbitrary.fp.v32f16.v6i32(<6 x i32> %x, metadata !"Float6E2M3FN")
  ret <32 x half> %r
}

; GFX950-LABEL: from_bf6_bf16:
; GFX950: v_cvt_scalef32_pk32_bf16_bf6 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 1.0
define <32 x bfloat> @from_bf6_bf16(<6 x i32> %x) {
  %r = call <32 x bfloat> @llvm.convert.from.arbitrary.fp.v32bf16.v6i32(<6 x i32> %x, metadata !"Float6E3M2FN")
  ret <32 x bfloat> %r
}

; GFX950-LABEL: to_fp6_f32:
; GFX950: v_cvt_scalef32_2xpk16_fp6_f32 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 1.0
define <6 x i32> @to_fp6_f32(<32 x float> %x) {
  %r = call <6 x i32> @llvm.convert.to.arbitrary.fp.v6i32.v32f32(<32 x float> %x, metadata !"Float6E2M3FN", metadata !"round.tonearest", i1 false)
  ret <6 x i32> %r
}

; GFX950-LABEL: to_bf6_f32:
; GFX950: v_cvt_scalef32_2xpk16_bf6_f32 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 1.0
define <6 x i32> @to_bf6_f32(<32 x float> %x) {
  %r = call <6 x i32> @llvm.convert.to.arbitrary.fp.v6i32.v32f32(<32 x float> %x, metadata !"Float6E3M2FN", metadata !"round.tonearest", i1 false)
  ret <6 x i32> %r
}

; GFX950-LABEL: to_fp6_f16:
; GFX950: v_cvt_scalef32_pk32_fp6_f16 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 1.0
define <6 x i32> @to_fp6_f16(<32 x half> %x) {
  %r = call <6 x i32> @llvm.convert.to.arbitrary.fp.v6i32.v32f16(<32 x half> %x, metadata !"Float6E2M3FN", metadata !"round.tonearest", i1 false)
  ret <6 x i32> %r
}

; GFX950-LABEL: to_bf6_bf16:
; GFX950: v_cvt_scalef32_pk32_bf6_bf16 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], 1.0
define <6 x i32> @to_bf6_bf16(<32 x bfloat> %x) {
  %r = call <6 x i32> @llvm.convert.to.arbitrary.fp.v6i32.v32bf16(<32 x bfloat> %x, metadata !"Float6E3M2FN", metadata !"round.tonearest", i1 false)
  ret <6 x i32> %r
}
