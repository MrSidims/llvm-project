; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_float8 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_float8 %s -o - -filetype=obj | spirv-val %}

; Test that llvm.convert.from.arbitrary.fp with an FP8 format lowers to the
; phantom-type OpBitcast + OpFConvert sequence.

; CHECK-DAG: OpCapability Float8EXT
; CHECK-DAG: OpExtension "SPV_EXT_float8"

; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Float32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FP8E4M3:]] = OpTypeFloat 8 4214
; CHECK-DAG: %[[#FP8E5M2:]] = OpTypeFloat 8 4215

declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)

; CHECK: %[[#ARG1:]] = OpFunctionParameter %[[#Int8]]
; CHECK: %[[#TMP1:]] = OpBitcast %[[#FP8E4M3]] %[[#ARG1]]
; CHECK: %[[#RES1:]] = OpFConvert %[[#Float32]] %[[#TMP1]]
; CHECK: OpReturnValue %[[#RES1]]
define float @from_e4m3(i8 %x) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %x, metadata !"Float8E4M3FN")
  ret float %r
}

; CHECK: %[[#ARG2:]] = OpFunctionParameter %[[#Int8]]
; CHECK: %[[#TMP2:]] = OpBitcast %[[#FP8E5M2]] %[[#ARG2]]
; CHECK: %[[#RES2:]] = OpFConvert %[[#Float32]] %[[#TMP2]]
; CHECK: OpReturnValue %[[#RES2]]
define float @from_e5m2(i8 %x) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %x, metadata !"Float8E5M2")
  ret float %r
}
