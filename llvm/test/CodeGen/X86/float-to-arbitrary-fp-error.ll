; RUN: split-file %s %t
; RUN: not llc < %t/float8e4m3.ll -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s --check-prefix=E4M3
; RUN: not llc < %t/float8e3m4.ll -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s --check-prefix=E3M4

; Test that llvm.convert.to.arbitrary.fp emits an error for formats that pass
; verifier validation but are not yet implemented in SelectionDAGBuilder.

;--- float8e4m3.ll
; E4M3: error: convert_to_arbitrary_fp: not implemented format 'Float8E4M3'

declare i8 @llvm.convert.to.arbitrary.fp.i8.f32(float, metadata, metadata, i1)

define i8 @to_f8e4m3(float %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(
      float %v, metadata !"Float8E4M3", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

;--- float8e3m4.ll
; E3M4: error: convert_to_arbitrary_fp: not implemented format 'Float8E3M4'

declare i8 @llvm.convert.to.arbitrary.fp.i8.f32(float, metadata, metadata, i1)

define i8 @to_f8e3m4(float %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(
      float %v, metadata !"Float8E3M4", metadata !"round.tonearest", i1 false)
  ret i8 %r
}
