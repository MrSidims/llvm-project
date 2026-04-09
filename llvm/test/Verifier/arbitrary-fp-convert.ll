;; Test verification of arbitrary FP conversion intrinsics:
;; - Metadata validation (interpretation, rounding mode)
;; - Type checking (pointer types, integer types, vector mismatches)
; RUN: split-file %s %t
; RUN: not llvm-as %t/bad-interpretation-empty.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BAD-INTERP-EMPTY
; RUN: not llvm-as %t/bad-interpretation-unknown.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BAD-INTERP-UNKNOWN
; RUN: not llvm-as %t/bad-rounding.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BAD-ROUNDING
; RUN: not opt -S -passes=verify %t/ptr-to-arbitrary-fp.ll 2>&1 | FileCheck %s --check-prefix=PTR-TO-FP
; RUN: not opt -S -passes=verify %t/arbitrary-fp-to-ptr.ll 2>&1 | FileCheck %s --check-prefix=FP-TO-PTR
; RUN: not opt -S -passes=verify %t/int-to-arbitrary-fp.ll 2>&1 | FileCheck %s --check-prefix=INT-TO-FP
; RUN: not opt -S -passes=verify %t/arbitrary-fp-to-int.ll 2>&1 | FileCheck %s --check-prefix=FP-TO-INT
; RUN: not opt -S -passes=verify %t/vec-ptr-to-arbitrary-fp.ll 2>&1 | FileCheck %s --check-prefix=VEC-PTR-TO-FP
; RUN: not opt -S -passes=verify %t/vec-to-scalar-mismatch.ll 2>&1 | FileCheck %s --check-prefix=VEC-SCALAR-MISMATCH
; RUN: not opt -S -passes=verify %t/vec-size-mismatch.ll 2>&1 | FileCheck %s --check-prefix=VEC-SIZE-MISMATCH
; RUN: not llvm-as %t/sr-bad-interpretation.ll -disable-output 2>&1 | FileCheck %s --check-prefix=SR-BAD-INTERP
; RUN: not opt -S -passes=verify %t/sr-size-mismatch.ll 2>&1 | FileCheck %s --check-prefix=SR-SIZE-MISMATCH

;--- bad-interpretation-empty.ll
; BAD-INTERP-EMPTY: interpretation metadata string must not be empty

declare i8 @llvm.convert.to.arbitrary.fp.i8.f16(half, metadata, metadata, i1)

define i8 @bad_interpretation_empty(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

;--- bad-interpretation-unknown.ll
; BAD-INTERP-UNKNOWN: unsupported interpretation metadata string

declare i8 @llvm.convert.to.arbitrary.fp.i8.f16(half, metadata, metadata, i1)

define i8 @bad_interpretation_unknown(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"unknown", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

;--- bad-rounding.ll
; BAD-ROUNDING: unsupported rounding mode argument

declare i8 @llvm.convert.to.arbitrary.fp.i8.f16(half, metadata, metadata, i1)

define i8 @bad_rounding(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"Float8E4M3", metadata !"round.dynamic", i1 false)
  ret i8 %r
}

;--- ptr-to-arbitrary-fp.ll
; PTR-TO-FP: Intrinsic has incorrect argument type!

declare i8 @llvm.convert.to.arbitrary.fp.i8.ptr(ptr, metadata, metadata, i1)

define i8 @bad_ptr_to_fp(ptr %p) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.ptr(
      ptr %p, metadata !"Float8E4M3", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

;--- arbitrary-fp-to-ptr.ll
; FP-TO-PTR: Intrinsic has incorrect return type!

declare ptr @llvm.convert.from.arbitrary.fp.ptr.i8(i8, metadata)

define ptr @bad_fp_to_ptr(i8 %v) {
  %r = call ptr @llvm.convert.from.arbitrary.fp.ptr.i8(
      i8 %v, metadata !"Float8E4M3")
  ret ptr %r
}

;--- int-to-arbitrary-fp.ll
; INT-TO-FP: Intrinsic has incorrect argument type!

declare i8 @llvm.convert.to.arbitrary.fp.i8.i32(i32, metadata, metadata, i1)

define i8 @bad_int_to_fp(i32 %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.i32(
      i32 %v, metadata !"Float8E4M3", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

;--- arbitrary-fp-to-int.ll
; FP-TO-INT: Intrinsic has incorrect return type!

declare i32 @llvm.convert.from.arbitrary.fp.i32.i8(i8, metadata)

define i32 @bad_fp_to_int(i8 %v) {
  %r = call i32 @llvm.convert.from.arbitrary.fp.i32.i8(
      i8 %v, metadata !"Float8E4M3")
  ret i32 %r
}

;--- vec-ptr-to-arbitrary-fp.ll
; VEC-PTR-TO-FP: Intrinsic has incorrect argument type!

declare <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4ptr(<4 x ptr>, metadata, metadata, i1)

define <4 x i8> @bad_vec_ptr_to_fp(<4 x ptr> %p) {
  %r = call <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4ptr(
      <4 x ptr> %p, metadata !"Float8E4M3", metadata !"round.tonearest", i1 false)
  ret <4 x i8> %r
}

;--- vec-to-scalar-mismatch.ll
; VEC-SCALAR-MISMATCH: integer operand total bit width must equal floating-point element count times the format's bits per element

declare i8 @llvm.convert.to.arbitrary.fp.i8.v4f16(<4 x half>, metadata, metadata, i1)

define i8 @bad_vec_to_scalar(<4 x half> %v) {
  ; <4 x half> with Float8E4M3 (8 bits/elt) needs 32 bits; i8 has only 8.
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.v4f16(
      <4 x half> %v, metadata !"Float8E4M3", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

;--- vec-size-mismatch.ll
; VEC-SIZE-MISMATCH: integer operand total bit width must equal floating-point element count times the format's bits per element

declare <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v2f32(<2 x float>, metadata, metadata, i1)

define <4 x i8> @bad_vec_size_mismatch(<2 x float> %v) {
  ; <2 x float> with Float8E4M3 (8 bits/elt) needs 16 bits; <4 x i8> is 32.
  %r = call <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v2f32(
      <2 x float> %v, metadata !"Float8E4M3", metadata !"round.tonearest", i1 false)
  ret <4 x i8> %r
}

;--- sr-bad-interpretation.ll
; SR-BAD-INTERP: unsupported interpretation metadata string

declare i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(float, metadata, i32, i1)

define i8 @sr_bad_interpretation(float %v, i32 %s) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.sr.i8.f32(
      float %v, metadata !"bogus", i32 %s, i1 false)
  ret i8 %r
}

;--- sr-size-mismatch.ll
; SR-SIZE-MISMATCH: integer operand total bit width must equal floating-point element count times the format's bits per element

declare i16 @llvm.convert.to.arbitrary.fp.sr.i16.f32(float, metadata, i32, i1)

define i16 @sr_size_mismatch(float %v, i32 %s) {
  ; f32 with Float4E2M1FN (4 bits/elt) needs 4 bits; i16 is 16.
  %r = call i16 @llvm.convert.to.arbitrary.fp.sr.i16.f32(
      float %v, metadata !"Float4E2M1FN", i32 %s, i1 false)
  ret i16 %r
}
