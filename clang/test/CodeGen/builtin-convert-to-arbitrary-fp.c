// RUN: %clang_cc1 -triple x86_64 -emit-llvm -o - %s | FileCheck %s

unsigned int test_e5m2(float x) {
  return __builtin_convert_to_arbitrary_fp(x, 0, 0, 0);
}
// CHECK-LABEL: @test_e5m2
// CHECK: call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float %{{.*}}, metadata !"Float8E5M2", metadata !"round.tonearest", i1 false)
// CHECK: zext i8 %{{.*}} to i32

unsigned int test_e4m3fn(float x) {
  return __builtin_convert_to_arbitrary_fp(x, 1, 0, 0);
}
// CHECK-LABEL: @test_e4m3fn
// CHECK: call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float %{{.*}}, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 false)
// CHECK: zext i8 %{{.*}} to i32

unsigned int test_e2m1fn(float x) {
  return __builtin_convert_to_arbitrary_fp(x, 2, 0, 0);
}
// CHECK-LABEL: @test_e2m1fn
// CHECK: call i4 @llvm.convert.to.arbitrary.fp.i4.f32(float %{{.*}}, metadata !"Float4E2M1FN", metadata !"round.tonearest", i1 false)
// CHECK: zext i4 %{{.*}} to i32

unsigned int test_e3m2fn(float x) {
  return __builtin_convert_to_arbitrary_fp(x, 3, 0, 0);
}
// CHECK-LABEL: @test_e3m2fn
// CHECK: call i6 @llvm.convert.to.arbitrary.fp.i6.f32(float %{{.*}}, metadata !"Float6E3M2FN", metadata !"round.tonearest", i1 false)
// CHECK: zext i6 %{{.*}} to i32

unsigned int test_e2m3fn(float x) {
  return __builtin_convert_to_arbitrary_fp(x, 4, 0, 0);
}
// CHECK-LABEL: @test_e2m3fn
// CHECK: call i6 @llvm.convert.to.arbitrary.fp.i6.f32(float %{{.*}}, metadata !"Float6E2M3FN", metadata !"round.tonearest", i1 false)
// CHECK: zext i6 %{{.*}} to i32

unsigned int test_round_towardzero(float x) {
  return __builtin_convert_to_arbitrary_fp(x, 0, 3, 0);
}
// CHECK-LABEL: @test_round_towardzero
// CHECK: call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float %{{.*}}, metadata !"Float8E5M2", metadata !"round.towardzero", i1 false)

unsigned int test_saturate(float x) {
  return __builtin_convert_to_arbitrary_fp(x, 1, 0, 1);
}
// CHECK-LABEL: @test_saturate
// CHECK: call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float %{{.*}}, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 true)
