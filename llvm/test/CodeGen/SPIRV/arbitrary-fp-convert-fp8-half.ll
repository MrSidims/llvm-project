; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_EXT_float8 %s -o - | FileCheck %s

; CHECK: OpExtension "SPV_EXT_float8"
; CHECK: OpCapability Float8EXT
; CHECK-DAG: OpDecorate %[[HALF_TO_FP8:[0-9]+]] FPRoundingMode RTZ
; CHECK-DAG: %[[HALF:[0-9]+]] = OpTypeFloat 16
; CHECK-DAG: %[[FLOAT:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[DOUBLE:[0-9]+]] = OpTypeFloat 64
; CHECK-DAG: %[[FP8E4:[0-9]+]] = OpTypeFloat 8 4214
; CHECK-DAG: %[[FP8E5:[0-9]+]] = OpTypeFloat 8 4215
; CHECK-DAG: %[[INT8:[0-9]+]] = OpTypeInt 8 {{[01]}}
; CHECK-DAG: %[[INT16:[0-9]+]] = OpTypeInt 16 {{[01]}}
; CHECK-DAG: %[[INT32:[0-9]+]] = OpTypeInt 32 {{[01]}}
; CHECK-DAG: %[[INT64:[0-9]+]] = OpTypeInt 64 {{[01]}}
; CHECK-DAG: %[[VHALF:[0-9]+]] = OpTypeVector %[[HALF]] 4
; CHECK-DAG: %[[VFP8:[0-9]+]] = OpTypeVector %[[FP8E4]] 4
; CHECK-DAG: %[[VINT8:[0-9]+]] = OpTypeVector %[[INT8]] 4
; CHECK-LABEL: %[[FROM_HALF:]] = OpFunction %[[HALF]] None
; CHECK: %[[ARG:]] = OpFunctionParameter %[[INT8]]
; CHECK: %[[FP8VAL:]] = OpBitcast %[[FP8E4]] %[[ARG]]
; CHECK: %[[HALFRES:]] = OpFConvert %[[HALF]] %[[FP8VAL]]
; CHECK: OpReturnValue %[[HALFRES]]
; CHECK-LABEL: %[[TO_HALF:]] = OpFunction %[[INT8]] None
; CHECK: %[[HALFARG:]] = OpFunctionParameter %[[HALF]]
; CHECK: %[[HALF_TO_FP8]] = OpFConvert %[[FP8E4]] %[[HALFARG]]
; CHECK: %[[BITCAST:]] = OpBitcast %[[INT8]] %[[HALF_TO_FP8]]
; CHECK: OpReturnValue %[[BITCAST]]
; CHECK-LABEL: %[[FROMVEC:]] = OpFunction %[[VHALF]] None
; CHECK: %[[VARG:]] = OpFunctionParameter %[[VINT8]]
; CHECK: %[[VFP8VAL:]] = OpBitcast %[[VFP8]] %[[VARG]]
; CHECK: %[[VHALFRES:]] = OpFConvert %[[VHALF]] %[[VFP8VAL]]
; CHECK: OpReturnValue %[[VHALFRES]]
; CHECK-LABEL: %[[TOVEC:]] = OpFunction %[[VINT8]] None
; CHECK: %[[VHALFARG:]] = OpFunctionParameter %[[VHALF]]
; CHECK: %[[VCONV:]] = OpFConvert %[[VFP8]] %[[VHALFARG]]
; CHECK: %[[VBITCAST:]] = OpBitcast %[[VINT8]] %[[VCONV]]
; CHECK: OpReturnValue %[[VBITCAST]]
; CHECK-LABEL: %[[FROM_FLOAT:]] = OpFunction %[[FLOAT]] None
; CHECK: %[[FARG:]] = OpFunctionParameter %[[INT8]]
; CHECK: %[[FFP8VAL:]] = OpBitcast %[[FP8E5]] %[[FARG]]
; CHECK: %[[FRES:]] = OpFConvert %[[FLOAT]] %[[FFP8VAL]]
; CHECK: OpReturnValue %[[FRES]]
; CHECK-LABEL: %[[TO_FLOAT:]] = OpFunction %[[INT8]] None
; CHECK: %[[FLARG:]] = OpFunctionParameter %[[FLOAT]]
; CHECK: %[[FLTCONV:]] = OpFConvert %[[FP8E5]] %[[FLARG]]
; CHECK: %[[FLBIT:]] = OpBitcast %[[INT8]] %[[FLTCONV]]
; CHECK: OpReturnValue %[[FLBIT]]
; CHECK-LABEL: %[[FROM_DOUBLE:]] = OpFunction %[[DOUBLE]] None
; CHECK: %[[DARG:]] = OpFunctionParameter %[[INT8]]
; CHECK: %[[DFP8VAL:]] = OpBitcast %[[FP8E5]] %[[DARG]]
; CHECK: %[[DRES:]] = OpFConvert %[[DOUBLE]] %[[DFP8VAL]]
; CHECK: OpReturnValue %[[DRES]]
; CHECK-LABEL: %[[TO_DOUBLE:]] = OpFunction %[[INT8]] None
; CHECK: %[[DLARG:]] = OpFunctionParameter %[[DOUBLE]]
; CHECK: %[[DLCONV:]] = OpFConvert %[[FP8E5]] %[[DLARG]]
; CHECK: %[[DLBIT:]] = OpBitcast %[[INT8]] %[[DLCONV]]
; CHECK: OpReturnValue %[[DLBIT]]
; CHECK-LABEL: %[[FP8_TO_I32:]] = OpFunction %[[INT32]] None
; CHECK: %[[I32SRC:]] = OpFunctionParameter %[[INT8]]
; CHECK: %[[I32FP8:]] = OpBitcast %[[FP8E4]] %[[I32SRC]]
; CHECK: %[[I32RES:]] = OpConvertFToS %[[INT32]] %[[I32FP8]]
; CHECK: OpReturnValue %[[I32RES]]
; CHECK-LABEL: %[[FP8_TO_U16:]] = OpFunction %[[INT16]] None
; CHECK: %[[U16SRC:]] = OpFunctionParameter %[[INT8]]
; CHECK: %[[U16FP8:]] = OpBitcast %[[FP8E5]] %[[U16SRC]]
; CHECK: %[[U16RES:]] = OpConvertFToU %[[INT16]] %[[U16FP8]]
; CHECK: OpReturnValue %[[U16RES]]
; CHECK-LABEL: %[[FP8_TO_I64:]] = OpFunction %[[INT64]] None
; CHECK: %[[I64SRC:]] = OpFunctionParameter %[[INT8]]
; CHECK: %[[I64FP8:]] = OpBitcast %[[FP8E5]] %[[I64SRC]]
; CHECK: %[[I64RES:]] = OpConvertFToS %[[INT64]] %[[I64FP8]]
; CHECK: OpReturnValue %[[I64RES]]
; CHECK-LABEL: %[[FP8_TO_I8:]] = OpFunction %[[INT8]] None
; CHECK: %[[I8SRC:]] = OpFunctionParameter %[[INT8]]
; CHECK: %[[I8FP8:]] = OpBitcast %[[FP8E4]] %[[I8SRC]]
; CHECK: %[[I8RES:]] = OpConvertFToS %[[INT8]] %[[I8FP8]]
; CHECK: OpReturnValue %[[I8RES]]
; CHECK-LABEL: %[[I32_TO_FP8_FN:]] = OpFunction %[[INT8]] None
; CHECK: %[[I32ARG:]] = OpFunctionParameter %[[INT32]]
; CHECK: %[[I32_TO_FP8]] = OpConvertSToF %[[FP8E4]] %[[I32ARG]]
; CHECK: %[[I32BIT:]] = OpBitcast %[[INT8]] %[[I32_TO_FP8]]
; CHECK: OpReturnValue %[[I32BIT]]
; CHECK-LABEL: %[[U16_TO_FP8_FN:]] = OpFunction %[[INT8]] None
; CHECK: %[[U16ARG:]] = OpFunctionParameter %[[INT16]]
; CHECK: %[[U16CONV:]] = OpConvertUToF %[[FP8E5]] %[[U16ARG]]
; CHECK: %[[U16BIT:]] = OpBitcast %[[INT8]] %[[U16CONV]]
; CHECK: OpReturnValue %[[U16BIT]]
; CHECK-LABEL: %[[I64_TO_FP8_FN:]] = OpFunction %[[INT8]] None
; CHECK: %[[I64ARG:]] = OpFunctionParameter %[[INT64]]
; CHECK: %[[I64CONV:]] = OpConvertSToF %[[FP8E5]] %[[I64ARG]]
; CHECK: %[[I64BIT:]] = OpBitcast %[[INT8]] %[[I64CONV]]
; CHECK: OpReturnValue %[[I64BIT]]
; CHECK-LABEL: %[[I8_TO_FP8_FN:]] = OpFunction %[[INT8]] None
; CHECK: %[[I8ARG:]] = OpFunctionParameter %[[INT8]]
; CHECK: %[[I8CONV:]] = OpConvertSToF %[[FP8E4]] %[[I8ARG]]
; CHECK: %[[I8BIT:]] = OpBitcast %[[INT8]] %[[I8CONV]]
; CHECK: OpReturnValue %[[I8BIT]]

declare half @llvm.arbitrary.fp.convert.half.i8(i8, metadata, metadata, metadata)
declare i8 @llvm.arbitrary.fp.convert.i8.half(half, metadata, metadata, metadata)
declare <4 x half> @llvm.arbitrary.fp.convert.v4f16.v4i8(<4 x i8>, metadata, metadata, metadata)
declare <4 x i8> @llvm.arbitrary.fp.convert.v4i8.v4f16(<4 x half>, metadata, metadata, metadata)
declare float @llvm.arbitrary.fp.convert.f32.i8(i8, metadata, metadata, metadata)
declare i8 @llvm.arbitrary.fp.convert.i8.f32(float, metadata, metadata, metadata)
declare double @llvm.arbitrary.fp.convert.f64.i8(i8, metadata, metadata, metadata)
declare i8 @llvm.arbitrary.fp.convert.i8.f64(double, metadata, metadata, metadata)
declare i32 @llvm.arbitrary.fp.convert.i32.i8(i8, metadata, metadata, metadata)
declare i16 @llvm.arbitrary.fp.convert.i16.i8(i8, metadata, metadata, metadata)
declare i64 @llvm.arbitrary.fp.convert.i64.i8(i8, metadata, metadata, metadata)
declare i8 @llvm.arbitrary.fp.convert.i8.i32(i32, metadata, metadata, metadata)
declare i8 @llvm.arbitrary.fp.convert.i8.i16(i16, metadata, metadata, metadata)
declare i8 @llvm.arbitrary.fp.convert.i8.i64(i64, metadata, metadata, metadata)
declare i8 @llvm.arbitrary.fp.convert.i8.i8(i8, metadata, metadata, metadata)

define half @from_fp8(i8 %v) {
  %r = call half @llvm.arbitrary.fp.convert.half.i8(
      i8 %v, metadata !"none", metadata !"Float8E4M3", metadata !"none")
  ret half %r
}

define i8 @to_fp8(half %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.half(
      half %v, metadata !"Float8E4M3", metadata !"none", metadata !"round.towardzero")
  ret i8 %r
}

define <4 x half> @from_fp8_vec(<4 x i8> %v) {
  %r = call <4 x half> @llvm.arbitrary.fp.convert.v4f16.v4i8(
      <4 x i8> %v, metadata !"none", metadata !"Float8E4M3", metadata !"none")
  ret <4 x half> %r
}

define <4 x i8> @to_fp8_vec(<4 x half> %v) {
  %r = call <4 x i8> @llvm.arbitrary.fp.convert.v4i8.v4f16(
      <4 x half> %v, metadata !"Float8E4M3", metadata !"none", metadata !"round.towardzero")
  ret <4 x i8> %r
}

define float @from_fp8_to_float(i8 %v) {
  %r = call float @llvm.arbitrary.fp.convert.f32.i8(
      i8 %v, metadata !"none", metadata !"Float8E5M2", metadata !"none")
  ret float %r
}

define i8 @to_fp8_from_float(float %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.f32(
      float %v, metadata !"Float8E5M2", metadata !"none", metadata !"round.towardzero")
  ret i8 %r
}

define double @from_fp8_to_double(i8 %v) {
  %r = call double @llvm.arbitrary.fp.convert.f64.i8(
      i8 %v, metadata !"none", metadata !"Float8E5M2", metadata !"none")
  ret double %r
}

define i8 @to_fp8_from_double(double %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.f64(
      double %v, metadata !"Float8E5M2", metadata !"none", metadata !"round.towardzero")
  ret i8 %r
}

define i32 @from_fp8_to_i32(i8 %v) {
  %r = call i32 @llvm.arbitrary.fp.convert.i32.i8(
      i8 %v, metadata !"signed", metadata !"Float8E4M3", metadata !"none")
  ret i32 %r
}

define i16 @from_fp8_to_u16(i8 %v) {
  %r = call i16 @llvm.arbitrary.fp.convert.i16.i8(
      i8 %v, metadata !"unsigned", metadata !"Float8E5M2", metadata !"none")
  ret i16 %r
}

define i8 @to_fp8_from_i32(i32 %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.i32(
      i32 %v, metadata !"Float8E4M3", metadata !"signed", metadata !"none")
  ret i8 %r
}

define i8 @to_fp8_from_u16(i16 %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.i16(
      i16 %v, metadata !"Float8E5M2", metadata !"unsigned", metadata !"none")
  ret i8 %r
}

define i64 @from_fp8_to_i64(i8 %v) {
  %r = call i64 @llvm.arbitrary.fp.convert.i64.i8(
      i8 %v, metadata !"signed", metadata !"Float8E5M2", metadata !"none")
  ret i64 %r
}

define i8 @from_fp8_to_i8(i8 %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.i8(
      i8 %v, metadata !"signed", metadata !"Float8E4M3", metadata !"none")
  ret i8 %r
}

define i8 @to_fp8_from_i64(i64 %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.i64(
      i64 %v, metadata !"Float8E5M2", metadata !"signed", metadata !"none")
  ret i8 %r
}

define i8 @to_fp8_from_i8(i8 %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.i8(
      i8 %v, metadata !"Float8E4M3", metadata !"signed", metadata !"none")
  ret i8 %r
}

