; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_float8 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_float8 %s -o - -filetype=obj | spirv-val %}

; Test that non-RNE rounding modes on llvm.convert.to.arbitrary.fp lower to
; an FPRoundingMode decoration on the OpFConvert result register. RTE is
; the default and emits no decoration; RTZ/RTP/RTN each emit one.

; CHECK-DAG: OpCapability Float8EXT
; CHECK-DAG: OpExtension "SPV_EXT_float8"

; Module-scope decorations (collected per function, all at the top):
; CHECK-DAG: OpDecorate %[[#RTZ_TMP:]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#RTP_TMP:]] FPRoundingMode RTP
; CHECK-DAG: OpDecorate %[[#RTN_TMP:]] FPRoundingMode RTN

; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Float32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FP8E4M3:]] = OpTypeFloat 8 4214

declare i8 @llvm.convert.to.arbitrary.fp.i8.f32(float, metadata, metadata, i1)

; RTE (default) must NOT emit an FPRoundingMode decoration. This check is
; implicit in the absence of an RTE entry in the DAG block above — FileCheck
; only matches decorations that are expected.
; CHECK: %[[#RTE_ARG:]] = OpFunctionParameter %[[#Float32]]
; CHECK: %[[#RTE_TMP:]] = OpFConvert %[[#FP8E4M3]] %[[#RTE_ARG]]
; CHECK: %[[#]] = OpBitcast %[[#Int8]] %[[#RTE_TMP]]
define i8 @to_e4m3_rte(float %x) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(
      float %x, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

; CHECK: %[[#RTZ_ARG:]] = OpFunctionParameter %[[#Float32]]
; CHECK: %[[#RTZ_TMP]] = OpFConvert %[[#FP8E4M3]] %[[#RTZ_ARG]]
; CHECK: %[[#]] = OpBitcast %[[#Int8]] %[[#RTZ_TMP]]
define i8 @to_e4m3_rtz(float %x) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(
      float %x, metadata !"Float8E4M3FN", metadata !"round.towardzero", i1 false)
  ret i8 %r
}

; CHECK: %[[#RTP_ARG:]] = OpFunctionParameter %[[#Float32]]
; CHECK: %[[#RTP_TMP]] = OpFConvert %[[#FP8E4M3]] %[[#RTP_ARG]]
; CHECK: %[[#]] = OpBitcast %[[#Int8]] %[[#RTP_TMP]]
define i8 @to_e4m3_rtp(float %x) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(
      float %x, metadata !"Float8E4M3FN", metadata !"round.upward", i1 false)
  ret i8 %r
}

; CHECK: %[[#RTN_ARG:]] = OpFunctionParameter %[[#Float32]]
; CHECK: %[[#RTN_TMP]] = OpFConvert %[[#FP8E4M3]] %[[#RTN_ARG]]
; CHECK: %[[#]] = OpBitcast %[[#Int8]] %[[#RTN_TMP]]
define i8 @to_e4m3_rtn(float %x) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(
      float %x, metadata !"Float8E4M3FN", metadata !"round.downward", i1 false)
  ret i8 %r
}
