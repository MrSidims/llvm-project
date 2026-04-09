; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_float8 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_float8 %s -o - -filetype=obj | spirv-val %}

; Test vector shapes of llvm.convert.{from,to}.arbitrary.fp for FP8. The
; selector must build the phantom OpTypeFloat 8 <enc> through
; getOrCreateSPIRVVectorTypeFromSPV to avoid the LLVM-Type round-trip
; required by the standard getOrCreateSPIRVVectorType path (the phantom
; type has no LLVM IR counterpart).

; CHECK-DAG: OpCapability Float8EXT
; CHECK-DAG: OpExtension "SPV_EXT_float8"

; The saturate-true TO-direction vector attaches the FP8 decoration to the
; OpFConvert result register at module scope.
; CHECK-DAG: OpDecorate %[[#SAT_TMP:]] SaturatedToLargestFloat8NormalConversionEXT

; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#V4Int8:]] = OpTypeVector %[[#Int8]] 4
; CHECK-DAG: %[[#Float32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#V4Float32:]] = OpTypeVector %[[#Float32]] 4
; CHECK-DAG: %[[#FP8E4M3:]] = OpTypeFloat 8 4214
; CHECK-DAG: %[[#V4FP8E4M3:]] = OpTypeVector %[[#FP8E4M3]] 4
; CHECK-DAG: %[[#FP8E5M2:]] = OpTypeFloat 8 4215
; CHECK-DAG: %[[#V4FP8E5M2:]] = OpTypeVector %[[#FP8E5M2]] 4

declare <4 x float> @llvm.convert.from.arbitrary.fp.v4f32.v4i8(<4 x i8>, metadata)
declare <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4f32(<4 x float>, metadata, metadata, i1)

; CHECK: %[[#ARG1:]] = OpFunctionParameter %[[#V4Int8]]
; CHECK: %[[#TMP1:]] = OpBitcast %[[#V4FP8E4M3]] %[[#ARG1]]
; CHECK: %[[#RES1:]] = OpFConvert %[[#V4Float32]] %[[#TMP1]]
; CHECK: OpReturnValue %[[#RES1]]
define <4 x float> @from_v4_e4m3(<4 x i8> %x) {
  %r = call <4 x float> @llvm.convert.from.arbitrary.fp.v4f32.v4i8(<4 x i8> %x, metadata !"Float8E4M3FN")
  ret <4 x float> %r
}

; CHECK: %[[#ARG2:]] = OpFunctionParameter %[[#V4Float32]]
; CHECK: %[[#TMP2:]] = OpFConvert %[[#V4FP8E4M3]] %[[#ARG2]]
; CHECK: %[[#RES2:]] = OpBitcast %[[#V4Int8]] %[[#TMP2]]
; CHECK: OpReturnValue %[[#RES2]]
define <4 x i8> @to_v4_e4m3_nosat(<4 x float> %x) {
  %r = call <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4f32(
      <4 x float> %x, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 false)
  ret <4 x i8> %r
}

; The saturated vector TO direction ties the decoration (asserted in the
; module-scope OpDecorate above) to the OpFConvert result register.
; CHECK: %[[#ARG3:]] = OpFunctionParameter %[[#V4Float32]]
; CHECK: %[[#SAT_TMP]] = OpFConvert %[[#V4FP8E5M2]] %[[#ARG3]]
; CHECK: %[[#RES3:]] = OpBitcast %[[#V4Int8]] %[[#SAT_TMP]]
; CHECK: OpReturnValue %[[#RES3]]
define <4 x i8> @to_v4_e5m2_saturate(<4 x float> %x) {
  %r = call <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4f32(
      <4 x float> %x, metadata !"Float8E5M2", metadata !"round.tonearest", i1 true)
  ret <4 x i8> %r
}
