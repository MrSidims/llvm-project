// RUN: %clang_cc1 -triple x86_64 -emit-llvm -o - %s | FileCheck %s

float test_e5m2(unsigned int x) {
  return __builtin_convert_from_arbitrary_fp(x, 0);
}
// CHECK-LABEL: @test_e5m2
// CHECK: %[[TRUNC:.*]] = trunc i32 %{{.*}} to i8
// CHECK: call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %[[TRUNC]], metadata !"Float8E5M2")

float test_e4m3fn(unsigned int x) {
  return __builtin_convert_from_arbitrary_fp(x, 1);
}
// CHECK-LABEL: @test_e4m3fn
// CHECK: %[[TRUNC:.*]] = trunc i32 %{{.*}} to i8
// CHECK: call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %[[TRUNC]], metadata !"Float8E4M3FN")

float test_e2m1fn(unsigned int x) {
  return __builtin_convert_from_arbitrary_fp(x, 2);
}
// CHECK-LABEL: @test_e2m1fn
// CHECK: %[[TRUNC:.*]] = trunc i32 %{{.*}} to i4
// CHECK: call float @llvm.convert.from.arbitrary.fp.f32.i4(i4 %[[TRUNC]], metadata !"Float4E2M1FN")

float test_e3m2fn(unsigned int x) {
  return __builtin_convert_from_arbitrary_fp(x, 3);
}
// CHECK-LABEL: @test_e3m2fn
// CHECK: %[[TRUNC:.*]] = trunc i32 %{{.*}} to i6
// CHECK: call float @llvm.convert.from.arbitrary.fp.f32.i6(i6 %[[TRUNC]], metadata !"Float6E3M2FN")

float test_e2m3fn(unsigned int x) {
  return __builtin_convert_from_arbitrary_fp(x, 4);
}
// CHECK-LABEL: @test_e2m3fn
// CHECK: %[[TRUNC:.*]] = trunc i32 %{{.*}} to i6
// CHECK: call float @llvm.convert.from.arbitrary.fp.f32.i6(i6 %[[TRUNC]], metadata !"Float6E2M3FN")
