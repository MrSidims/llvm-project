; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_float4,+SPV_INTEL_int4 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_float4,+SPV_INTEL_int4 %s -o - -filetype=obj | spirv-val %}

; Test llvm.convert.{from,to}.arbitrary.fp for Float4E2M1FN. The lowering
; requires both SPV_INTEL_float4 (the FP4 type) and SPV_INTEL_int4 (the i4
; carrier, which would otherwise be widened to i8 by SPIRVPreLegalizer).

; CHECK-DAG: OpCapability Float4E2M1TypeINTEL
; CHECK-DAG: OpCapability Int4TypeINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_float4"
; CHECK-DAG: OpExtension "SPV_INTEL_int4"

; CHECK-DAG: %[[#Float32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FP4:]] = OpTypeFloat 4 6214

declare float @llvm.convert.from.arbitrary.fp.f32.i4(i4, metadata)
declare i4 @llvm.convert.to.arbitrary.fp.i4.f32(float, metadata, metadata, i1)

; CHECK: %[[#TMP1:]] = OpBitcast %[[#FP4]] %[[#]]
; CHECK: %[[#RES1:]] = OpFConvert %[[#Float32]] %[[#TMP1]]
define float @from_e2m1(i4 %x) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i4(i4 %x, metadata !"Float4E2M1FN")
  ret float %r
}

; CHECK: %[[#TMP2:]] = OpFConvert %[[#FP4]] %[[#]]
; CHECK: %[[#RES2:]] = OpBitcast %[[#]] %[[#TMP2]]
define i4 @to_e2m1(float %x) {
  %r = call i4 @llvm.convert.to.arbitrary.fp.i4.f32(
      float %x, metadata !"Float4E2M1FN", metadata !"round.tonearest", i1 false)
  ret i4 %r
}
