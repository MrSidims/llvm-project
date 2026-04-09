; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_float8,+SPV_INTEL_fp_conversions %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_float8,+SPV_INTEL_fp_conversions %s -o - -filetype=obj | spirv-val %}

; Test that llvm.convert.to.arbitrary.fp.sr lowers to
; OpStochasticRoundFToFINTEL (saturate=false) or
; OpClampStochasticRoundFToFINTEL (saturate=true).

; CHECK-DAG: OpCapability FloatConversionsINTEL
; CHECK-DAG: OpCapability Float8EXT
; CHECK-DAG: OpExtension "SPV_EXT_float8"
; CHECK-DAG: OpExtension "SPV_INTEL_fp_conversions"

; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Float32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FP8E4M3:]] = OpTypeFloat 8 4214

declare i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float, metadata, i32, i1)

; CHECK: %[[#SRC1:]] = OpFunctionParameter %[[#Float32]]
; CHECK: %[[#SEED1:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#TMP1:]] = OpStochasticRoundFToFINTEL %[[#FP8E4M3]] %[[#SRC1]] %[[#SEED1]]
; CHECK: %[[#RES1:]] = OpBitcast %[[#Int8]] %[[#TMP1]]
define i8 @sr_e4m3_nosat(float %x, i32 %seed) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(
      float %x, metadata !"Float8E4M3FN", i32 %seed, i1 false)
  ret i8 %r
}

; CHECK: %[[#SRC2:]] = OpFunctionParameter %[[#Float32]]
; CHECK: %[[#SEED2:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#TMP2:]] = OpClampStochasticRoundFToFINTEL %[[#FP8E4M3]] %[[#SRC2]] %[[#SEED2]]
; CHECK: %[[#RES2:]] = OpBitcast %[[#Int8]] %[[#TMP2]]
define i8 @sr_e4m3_saturate(float %x, i32 %seed) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(
      float %x, metadata !"Float8E4M3FN", i32 %seed, i1 true)
  ret i8 %r
}
