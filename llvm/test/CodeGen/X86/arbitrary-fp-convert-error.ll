; RUN: split-file %s %t
; RUN: not llc < %t/float8e4m3.ll -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s --check-prefix=E4M3
; RUN: not llc < %t/float8e3m4.ll -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s --check-prefix=E3M4

; Test that llvm.convert.from.arbitrary.fp emits an error for formats that pass
; verifier validation but are not yet implemented in SelectionDAGBuilder.

;--- float8e4m3.ll
; E4M3: error: convert_from_arbitrary_fp: not implemented format 'Float8E4M3'

declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)

define float @from_f8e4m3(i8 %v) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(
      i8 %v, metadata !"Float8E4M3")
  ret float %r
}

;--- float8e3m4.ll
; E3M4: error: convert_from_arbitrary_fp: not implemented format 'Float8E3M4'

declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)

define float @from_f8e3m4(i8 %v) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(
      i8 %v, metadata !"Float8E3M4")
  ret float %r
}
