; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_float4,+SPV_INTEL_int4,+SPV_INTEL_fp_conversions %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_float4,+SPV_INTEL_int4,+SPV_INTEL_fp_conversions %s -o - -filetype=obj | spirv-val %}

; Test llvm.convert.to.arbitrary.fp.sr for Float4E2M1FN. Both saturate=false
; (OpStochasticRoundFToFINTEL) and saturate=true (OpClampStochasticRoundFToFINTEL)
; paths produce a phantom OpTypeFloat 4 Float4E2M1INTEL result that is then
; bitcast back to the i4 carrier.

; CHECK-DAG: OpCapability Float4E2M1TypeINTEL
; CHECK-DAG: OpCapability Int4TypeINTEL
; CHECK-DAG: OpCapability FloatConversionsINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_float4"
; CHECK-DAG: OpExtension "SPV_INTEL_int4"
; CHECK-DAG: OpExtension "SPV_INTEL_fp_conversions"

; CHECK-DAG: %[[#Float32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Int4:]] = OpTypeInt 4 0
; CHECK-DAG: %[[#FP4:]] = OpTypeFloat 4 6214

declare i4 @llvm.convert.to.arbitrary.fp.sr.i4.f32(float, metadata, i32, i1)

; CHECK: %[[#SRC1:]] = OpFunctionParameter %[[#Float32]]
; CHECK: %[[#SEED1:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#TMP1:]] = OpStochasticRoundFToFINTEL %[[#FP4]] %[[#SRC1]] %[[#SEED1]]
; CHECK: %[[#RES1:]] = OpBitcast %[[#Int4]] %[[#TMP1]]
define i4 @sr_fp4_nosat(float %x, i32 %seed) {
  %r = call i4 @llvm.convert.to.arbitrary.fp.sr.i4.f32(
      float %x, metadata !"Float4E2M1FN", i32 %seed, i1 false)
  ret i4 %r
}

; CHECK: %[[#SRC2:]] = OpFunctionParameter %[[#Float32]]
; CHECK: %[[#SEED2:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#TMP2:]] = OpClampStochasticRoundFToFINTEL %[[#FP4]] %[[#SRC2]] %[[#SEED2]]
; CHECK: %[[#RES2:]] = OpBitcast %[[#Int4]] %[[#TMP2]]
define i4 @sr_fp4_saturate(float %x, i32 %seed) {
  %r = call i4 @llvm.convert.to.arbitrary.fp.sr.i4.f32(
      float %x, metadata !"Float4E2M1FN", i32 %seed, i1 true)
  ret i4 %r
}
