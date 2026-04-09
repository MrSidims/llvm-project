; Negative tests for the SPIR-V lowering of llvm.convert.{from,to}.arbitrary.fp:
; unsupported formats and missing extension flags should produce clean errors.

; RUN: split-file %s %t
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_float8 \
; RUN:   %t/unsupported-e8m0.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E8M0
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_float8 \
; RUN:   %t/unsupported-fnuz.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FNUZ
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   %t/missing-ext.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOEXT

; E8M0 (exponent-only) and FNUZ FP8 variants have no SPIR-V encoding. FP6
; isn't covered here because its bit width doesn't divide any standard
; integer type cleanly, so the IR verifier rejects the minimal form before
; the SPIR-V backend ever sees it. `round.dynamic` is rejected by the IR
; verifier on the generic intrinsic before SPIR-V emission runs; the
; supported rounding modes (round.tonearest, round.towardzero,
; round.upward, round.downward) are all routed through FPRoundingMode
; decorations in the SPIR-V path.

; E8M0: SPIR-V has no encoding for arbitrary-fp format 'Float8E8M0FNU'
; FNUZ: SPIR-V has no encoding for arbitrary-fp format 'Float8E4M3FNUZ'
; NOEXT: requires the following SPIR-V extension: SPV_EXT_float8

;--- unsupported-e8m0.ll
declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)
define float @bad(i8 %x) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %x, metadata !"Float8E8M0FNU")
  ret float %r
}

;--- unsupported-fnuz.ll
declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)
define float @bad(i8 %x) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %x, metadata !"Float8E4M3FNUZ")
  ret float %r
}

;--- missing-ext.ll
declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)
define float @bad(i8 %x) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %x, metadata !"Float8E4M3FN")
  ret float %r
}
