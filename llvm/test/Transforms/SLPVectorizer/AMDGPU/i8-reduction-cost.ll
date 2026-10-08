; RUN: opt -passes=slp-vectorizer -mtriple=amdgpu9.00-amd-amdhsa -pass-remarks=slp-vectorizer -disable-output < %s 2>&1 | FileCheck -check-prefix=GFX9 %s
; RUN: opt -passes=slp-vectorizer -mtriple=amdgpu11.00-amd-amdhsa -pass-remarks=slp-vectorizer -pass-remarks-missed=slp-vectorizer -disable-output < %s 2>&1 | FileCheck -check-prefix=GFX11 %s

; GFX9: Vectorized horizontal reduction with cost -14 and with tree size 3
; GFX11: Vectorizing horizontal reduction is possible but not beneficial with cost 11 and threshold 0
; GFX11-NOT: Vectorized horizontal reduction
define i8 @add_mul_i8(ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %pa0 = getelementptr i8, ptr addrspace(1) %a, i64 0
  %pb0 = getelementptr i8, ptr addrspace(1) %b, i64 0
  %x0 = load i8, ptr addrspace(1) %pa0
  %y0 = load i8, ptr addrspace(1) %pb0
  %m0 = mul i8 %x0, %y0
  %pa1 = getelementptr i8, ptr addrspace(1) %a, i64 1
  %pb1 = getelementptr i8, ptr addrspace(1) %b, i64 1
  %x1 = load i8, ptr addrspace(1) %pa1
  %y1 = load i8, ptr addrspace(1) %pb1
  %m1 = mul i8 %x1, %y1
  %pa2 = getelementptr i8, ptr addrspace(1) %a, i64 2
  %pb2 = getelementptr i8, ptr addrspace(1) %b, i64 2
  %x2 = load i8, ptr addrspace(1) %pa2
  %y2 = load i8, ptr addrspace(1) %pb2
  %m2 = mul i8 %x2, %y2
  %pa3 = getelementptr i8, ptr addrspace(1) %a, i64 3
  %pb3 = getelementptr i8, ptr addrspace(1) %b, i64 3
  %x3 = load i8, ptr addrspace(1) %pa3
  %y3 = load i8, ptr addrspace(1) %pb3
  %m3 = mul i8 %x3, %y3
  %pa4 = getelementptr i8, ptr addrspace(1) %a, i64 4
  %pb4 = getelementptr i8, ptr addrspace(1) %b, i64 4
  %x4 = load i8, ptr addrspace(1) %pa4
  %y4 = load i8, ptr addrspace(1) %pb4
  %m4 = mul i8 %x4, %y4
  %pa5 = getelementptr i8, ptr addrspace(1) %a, i64 5
  %pb5 = getelementptr i8, ptr addrspace(1) %b, i64 5
  %x5 = load i8, ptr addrspace(1) %pa5
  %y5 = load i8, ptr addrspace(1) %pb5
  %m5 = mul i8 %x5, %y5
  %pa6 = getelementptr i8, ptr addrspace(1) %a, i64 6
  %pb6 = getelementptr i8, ptr addrspace(1) %b, i64 6
  %x6 = load i8, ptr addrspace(1) %pa6
  %y6 = load i8, ptr addrspace(1) %pb6
  %m6 = mul i8 %x6, %y6
  %pa7 = getelementptr i8, ptr addrspace(1) %a, i64 7
  %pb7 = getelementptr i8, ptr addrspace(1) %b, i64 7
  %x7 = load i8, ptr addrspace(1) %pa7
  %y7 = load i8, ptr addrspace(1) %pb7
  %m7 = mul i8 %x7, %y7
  %r1 = add i8 %m0, %m1
  %r2 = add i8 %r1, %m2
  %r3 = add i8 %r2, %m3
  %r4 = add i8 %r3, %m4
  %r5 = add i8 %r4, %m5
  %r6 = add i8 %r5, %m6
  %r7 = add i8 %r6, %m7
  ret i8 %r7
}
