; RUN: opt -passes=slp-vectorizer -verify-each -S -mtriple=x86_64-unknown-linux-gnu -mcpu=haswell < %s | FileCheck %s
; RUN: opt -passes=slp-vectorizer -verify-each -S -mtriple=x86_64-unknown-linux-gnu -mcpu=znver4 < %s | FileCheck %s
; RUN: opt -passes=slp-vectorizer -verify-each -S -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake-avx512 < %s | FileCheck %s

; These CPUs promote half arithmetic. A cheap promoted fmuladd intrinsic does
; not mean that separate half multiplies/adds will fuse: their intermediate
; rounding remains observable. Do not invent a scalar FMA saving and prevent
; vectorization of the conversions and multiplies.

; CHECK-LABEL: define half @half_contract(
; CHECK: fmul contract <4 x half>
; CHECK: ret half
define half @half_contract(ptr noalias readonly %a, ptr noalias readonly %b) {
  %ap0 = getelementptr half, ptr %a, i64 0
  %a0 = load half, ptr %ap0, align 2
  %bp0 = getelementptr half, ptr %b, i64 0
  %b0 = load half, ptr %bp0, align 2
  %m0 = fmul contract half %a0, %b0
  %ap1 = getelementptr half, ptr %a, i64 1
  %a1 = load half, ptr %ap1, align 2
  %bp1 = getelementptr half, ptr %b, i64 1
  %b1 = load half, ptr %bp1, align 2
  %m1 = fmul contract half %a1, %b1
  %r1 = fadd contract half %m0, %m1
  %ap2 = getelementptr half, ptr %a, i64 2
  %a2 = load half, ptr %ap2, align 2
  %bp2 = getelementptr half, ptr %b, i64 2
  %b2 = load half, ptr %bp2, align 2
  %m2 = fmul contract half %a2, %b2
  %r2 = fadd contract half %r1, %m2
  %ap3 = getelementptr half, ptr %a, i64 3
  %a3 = load half, ptr %ap3, align 2
  %bp3 = getelementptr half, ptr %b, i64 3
  %b3 = load half, ptr %bp3, align 2
  %m3 = fmul contract half %a3, %b3
  %r3 = fadd contract half %r2, %m3
  ret half %r3
}

; CHECK-LABEL: define half @half_contract_optsize(
; CHECK: fmul contract <4 x half>
; CHECK: ret half
define half @half_contract_optsize(ptr noalias readonly %a, ptr noalias readonly %b) optsize {
  %ap0 = getelementptr half, ptr %a, i64 0
  %a0 = load half, ptr %ap0, align 2
  %bp0 = getelementptr half, ptr %b, i64 0
  %b0 = load half, ptr %bp0, align 2
  %m0 = fmul contract half %a0, %b0
  %ap1 = getelementptr half, ptr %a, i64 1
  %a1 = load half, ptr %ap1, align 2
  %bp1 = getelementptr half, ptr %b, i64 1
  %b1 = load half, ptr %bp1, align 2
  %m1 = fmul contract half %a1, %b1
  %r1 = fadd contract half %m0, %m1
  %ap2 = getelementptr half, ptr %a, i64 2
  %a2 = load half, ptr %ap2, align 2
  %bp2 = getelementptr half, ptr %b, i64 2
  %b2 = load half, ptr %bp2, align 2
  %m2 = fmul contract half %a2, %b2
  %r2 = fadd contract half %r1, %m2
  %ap3 = getelementptr half, ptr %a, i64 3
  %a3 = load half, ptr %ap3, align 2
  %bp3 = getelementptr half, ptr %b, i64 3
  %b3 = load half, ptr %bp3, align 2
  %m3 = fmul contract half %a3, %b3
  %r3 = fadd contract half %r2, %m3
  ret half %r3
}

; CHECK-LABEL: define half @half_no_contract(
; CHECK: fmul <4 x half>
; CHECK: ret half
define half @half_no_contract(ptr noalias readonly %a, ptr noalias readonly %b) {
  %ap0 = getelementptr half, ptr %a, i64 0
  %a0 = load half, ptr %ap0, align 2
  %bp0 = getelementptr half, ptr %b, i64 0
  %b0 = load half, ptr %bp0, align 2
  %m0 = fmul half %a0, %b0
  %ap1 = getelementptr half, ptr %a, i64 1
  %a1 = load half, ptr %ap1, align 2
  %bp1 = getelementptr half, ptr %b, i64 1
  %b1 = load half, ptr %bp1, align 2
  %m1 = fmul half %a1, %b1
  %r1 = fadd half %m0, %m1
  %ap2 = getelementptr half, ptr %a, i64 2
  %a2 = load half, ptr %ap2, align 2
  %bp2 = getelementptr half, ptr %b, i64 2
  %b2 = load half, ptr %bp2, align 2
  %m2 = fmul half %a2, %b2
  %r2 = fadd half %r1, %m2
  %ap3 = getelementptr half, ptr %a, i64 3
  %a3 = load half, ptr %ap3, align 2
  %bp3 = getelementptr half, ptr %b, i64 3
  %b3 = load half, ptr %bp3, align 2
  %m3 = fmul half %a3, %b3
  %r3 = fadd half %r2, %m3
  ret half %r3
}
