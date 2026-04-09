; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_float8 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_float8 %s -o - -filetype=obj | spirv-val %}

; Test that llvm.convert.to.arbitrary.fp with an FP8 format lowers to the
; OpFConvert + OpBitcast sequence, and that `saturate=true` attaches the
; SaturatedToLargestFloat8NormalConversionEXT decoration to the OpFConvert
; result register.

; CHECK-DAG: OpCapability Float8EXT
; CHECK-DAG: OpExtension "SPV_EXT_float8"

; The decoration only fires on the saturating function's temporary, not the
; non-saturating one.
; CHECK-DAG: OpDecorate %[[#SAT_TMP:]] SaturatedToLargestFloat8NormalConversionEXT

; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Float32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FP8E4M3:]] = OpTypeFloat 8 4214

declare i8 @llvm.convert.to.arbitrary.fp.i8.f32(float, metadata, metadata, i1)

; CHECK: %[[#ARG1:]] = OpFunctionParameter %[[#Float32]]
; CHECK: %[[#NOSAT_TMP:]] = OpFConvert %[[#FP8E4M3]] %[[#ARG1]]
; CHECK: %[[#RES1:]] = OpBitcast %[[#Int8]] %[[#NOSAT_TMP]]
; CHECK: OpReturnValue %[[#RES1]]
define i8 @to_e4m3_nosat(float %x) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(
      float %x, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

; CHECK: %[[#ARG2:]] = OpFunctionParameter %[[#Float32]]
; CHECK: %[[#SAT_TMP]] = OpFConvert %[[#FP8E4M3]] %[[#ARG2]]
; CHECK: %[[#RES2:]] = OpBitcast %[[#Int8]] %[[#SAT_TMP]]
; CHECK: OpReturnValue %[[#RES2]]
define i8 @to_e4m3_saturate(float %x) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(
      float %x, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 true)
  ret i8 %r
}
