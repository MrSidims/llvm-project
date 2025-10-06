; RUN: split-file %s %t
; RUN: not opt -S -passes=verify %t/bad-result.ll 2>&1 | FileCheck %s --check-prefix=BADRESULT
; RUN: not opt -S -passes=verify %t/bad-rounding.ll 2>&1 | FileCheck %s --check-prefix=BADROUND
; RUN: not opt -S -passes=verify %t/bad-int-rounding.ll 2>&1 | FileCheck %s --check-prefix=BADINTRND
; RUN: not opt -S -passes=verify %t/bad-input-rounding.ll 2>&1 | FileCheck %s --check-prefix=BADINPRND
; RUN: opt -S -passes=verify %t/good.ll

;--- bad-result.ll
; BADRESULT: unsupported result interpretation metadata string
declare half @llvm.arbitrary.fp.convert.half.i8(i8, metadata, metadata, metadata)

define half @bad_result(i8 %v) {
  %r = call half @llvm.arbitrary.fp.convert.half.i8(
      i8 %v, metadata !"bogus", metadata !"Float8E5M2", metadata !"none")
  ret half %r
}

;--- bad-rounding.ll
; BADROUND: unsupported rounding mode argument
declare i8 @llvm.arbitrary.fp.convert.i8.half(half, metadata, metadata, metadata)

define i8 @bad_rounding(half %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.half(
      half %v, metadata !"Float8E4M3", metadata !"none", metadata !"round.dynamic")
  ret i8 %r
}

;--- bad-int-rounding.ll
; BADINTRND: rounding mode must be "none" when result interpretation is "none" or integer
declare i32 @llvm.arbitrary.fp.convert.i32.i8(i8, metadata, metadata, metadata)

define i32 @bad_int_rounding(i8 %v) {
  %r = call i32 @llvm.arbitrary.fp.convert.i32.i8(
      i8 %v, metadata !"signed", metadata !"Float8E4M3", metadata !"round.towardzero")
  ret i32 %r
}

;--- bad-input-rounding.ll
; BADINPRND: rounding mode must be "none" when input interpretation is integer
declare i8 @llvm.arbitrary.fp.convert.i8.i32(i32, metadata, metadata, metadata)

define i8 @bad_input_rounding(i32 %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.i32(
      i32 %v, metadata !"Float8E4M3", metadata !"signed", metadata !"round.towardzero")
  ret i8 %r
}

;--- good.ll
declare half @llvm.arbitrary.fp.convert.half.i8(i8, metadata, metadata, metadata)
declare i8 @llvm.arbitrary.fp.convert.i8.half(half, metadata, metadata, metadata)

define half @good_from(i8 %v) {
  %r = call half @llvm.arbitrary.fp.convert.half.i8(
      i8 %v, metadata !"none", metadata !"Float8E4M3", metadata !"none")
  ret half %r
}

define i8 @good_to(half %v) {
  %r = call i8 @llvm.arbitrary.fp.convert.i8.half(
      half %v, metadata !"Float8E4M3", metadata !"none", metadata !"round.towardzero")
  ret i8 %r
}
