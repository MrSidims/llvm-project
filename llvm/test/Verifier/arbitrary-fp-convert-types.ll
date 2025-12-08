;; Check that pointer types and incorrect type combinations are rejected
;; by the arbitrary FP conversion intrinsics at verification time
; RUN: split-file %s %t
; RUN: not opt -S -passes=verify %t/ptr-to-arbitrary-fp.ll 2>&1 | FileCheck %s --check-prefix=PTR-TO-FP
; RUN: not opt -S -passes=verify %t/arbitrary-fp-to-ptr.ll 2>&1 | FileCheck %s --check-prefix=FP-TO-PTR
; RUN: not opt -S -passes=verify %t/int-to-arbitrary-fp.ll 2>&1 | FileCheck %s --check-prefix=INT-TO-FP
; RUN: not opt -S -passes=verify %t/arbitrary-fp-to-int.ll 2>&1 | FileCheck %s --check-prefix=FP-TO-INT
; RUN: not opt -S -passes=verify %t/vec-ptr-to-arbitrary-fp.ll 2>&1 | FileCheck %s --check-prefix=VEC-PTR-TO-FP
; RUN: not opt -S -passes=verify %t/vec-to-scalar-mismatch.ll 2>&1 | FileCheck %s --check-prefix=VEC-SCALAR-MISMATCH
; RUN: not opt -S -passes=verify %t/vec-size-mismatch.ll 2>&1 | FileCheck %s --check-prefix=VEC-SIZE-MISMATCH

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

declare ptr @llvm.convert.from.arbitrary.fp.ptr.i8(i8, metadata, metadata, i1)

define ptr @bad_fp_to_ptr(i8 %v) {
  %r = call ptr @llvm.convert.from.arbitrary.fp.ptr.i8(
      i8 %v, metadata !"Float8E4M3", metadata !"round.tonearest", i1 false)
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

declare i32 @llvm.convert.from.arbitrary.fp.i32.i8(i8, metadata, metadata, i1)

define i32 @bad_fp_to_int(i8 %v) {
  %r = call i32 @llvm.convert.from.arbitrary.fp.i32.i8(
      i8 %v, metadata !"Float8E4M3", metadata !"round.tonearest", i1 false)
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
; VEC-SCALAR-MISMATCH: if floating-point operand is a vector, integer operand must also be a vector

declare i8 @llvm.convert.to.arbitrary.fp.i8.v4f16(<4 x half>, metadata, metadata, i1)

define i8 @bad_vec_to_scalar(<4 x half> %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.v4f16(
      <4 x half> %v, metadata !"Float8E4M3", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

;--- vec-size-mismatch.ll
; VEC-SIZE-MISMATCH: floating-point and integer vector operands must have the same element count

declare <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v2f32(<2 x float>, metadata, metadata, i1)

define <4 x i8> @bad_vec_size_mismatch(<2 x float> %v) {
  %r = call <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v2f32(
      <2 x float> %v, metadata !"Float8E4M3", metadata !"round.tonearest", i1 false)
  ret <4 x i8> %r
}
