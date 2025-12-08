; RUN: split-file %s %t
; RUN: not llvm-as %t/bad-interpretation-empty.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BAD-INTERP-EMPTY
; RUN: not llvm-as %t/bad-interpretation-unknown.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BAD-INTERP-UNKNOWN
; RUN: not llvm-as %t/bad-rounding.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BAD-ROUNDING

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
