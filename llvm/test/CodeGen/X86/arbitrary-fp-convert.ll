; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; Test conversion from Float4E2M1FN (FP4) to f16
define half @fp4_to_f16(i4 %val) {
; CHECK-LABEL: fp4_to_f16:
; CHECK:       # %bb.0:
; CHECK:       retq
  %result = call half @llvm.convert.from.arbitrary.fp.f16.i4(i4 %val, metadata !"Float4E2M1FN")
  ret half %result
}

; Test conversion from f16 to Float4E2M1FN (FP4)
define i4 @f16_to_fp4(half %val) {
; CHECK-LABEL: f16_to_fp4:
; CHECK:       # %bb.0:
; CHECK:       retq
  %result = call i4 @llvm.convert.to.arbitrary.fp.i4.f16(half %val, metadata !"Float4E2M1FN", metadata !"round.tonearest", i1 false)
  ret i4 %result
}

; Test conversion from Float8E4M3FN (FP8) to f16
define half @fp8_to_f16(i8 %val) {
; CHECK-LABEL: fp8_to_f16:
; CHECK:       # %bb.0:
; CHECK:       retq
  %result = call half @llvm.convert.from.arbitrary.fp.f16.i8(i8 %val, metadata !"Float8E4M3FN")
  ret half %result
}

; Test conversion from f16 to Float8E4M3FN (FP8)
define i8 @f16_to_fp8(half %val) {
; CHECK-LABEL: f16_to_fp8:
; CHECK:       # %bb.0:
; CHECK:       retq
  %result = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(half %val, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 false)
  ret i8 %result
}

; Test conversion with saturation flag
define i4 @f16_to_fp4_saturate(half %val) {
; CHECK-LABEL: f16_to_fp4_saturate:
; CHECK:       # %bb.0:
; CHECK:       retq
  %result = call i4 @llvm.convert.to.arbitrary.fp.i4.f16(half %val, metadata !"Float4E2M1FN", metadata !"round.tonearest", i1 true)
  ret i4 %result
}

; Test conversion with different rounding modes
define i4 @f16_to_fp4_round_zero(half %val) {
; CHECK-LABEL: f16_to_fp4_round_zero:
; CHECK:       # %bb.0:
; CHECK:       retq
  %result = call i4 @llvm.convert.to.arbitrary.fp.i4.f16(half %val, metadata !"Float4E2M1FN", metadata !"round.towardzero", i1 false)
  ret i4 %result
}

; Test conversion to f32 (should go FP4 -> FP16 -> FP32)
define float @fp4_to_f32(i4 %val) {
; CHECK-LABEL: fp4_to_f32:
; CHECK:       # %bb.0:
; CHECK:       retq
  %result = call float @llvm.convert.from.arbitrary.fp.f32.i4(i4 %val, metadata !"Float4E2M1FN")
  ret float %result
}

; Test conversion from f32 to FP4 (should go FP32 -> FP16 -> FP4)
define i4 @f32_to_fp4(float %val) {
; CHECK-LABEL: f32_to_fp4:
; CHECK:       # %bb.0:
; CHECK:       retq
  %result = call i4 @llvm.convert.to.arbitrary.fp.i4.f32(float %val, metadata !"Float4E2M1FN", metadata !"round.tonearest", i1 false)
  ret i4 %result
}

; Test conversion from FP8 to f32 (should go FP8 -> FP16 -> FP32)
define float @fp8_to_f32(i8 %val) {
; CHECK-LABEL: fp8_to_f32:
; CHECK:       # %bb.0:
; CHECK:       retq
  %result = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %val, metadata !"Float8E4M3FN")
  ret float %result
}

; Test conversion from f32 to FP8 (should go FP32 -> FP16 -> FP8)
define i8 @f32_to_fp8(float %val) {
; CHECK-LABEL: f32_to_fp8:
; CHECK:       # %bb.0:
; CHECK:       retq
  %result = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float %val, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 false)
  ret i8 %result
}

; Declare the intrinsics
declare half @llvm.convert.from.arbitrary.fp.f16.i4(i4, metadata)
declare half @llvm.convert.from.arbitrary.fp.f16.i8(i8, metadata)
declare float @llvm.convert.from.arbitrary.fp.f32.i4(i4, metadata)
declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)
declare i4 @llvm.convert.to.arbitrary.fp.i4.f16(half, metadata, metadata, i1)
declare i8 @llvm.convert.to.arbitrary.fp.i8.f16(half, metadata, metadata, i1)
declare i4 @llvm.convert.to.arbitrary.fp.i4.f32(float, metadata, metadata, i1)
declare i8 @llvm.convert.to.arbitrary.fp.i8.f32(float, metadata, metadata, i1)
