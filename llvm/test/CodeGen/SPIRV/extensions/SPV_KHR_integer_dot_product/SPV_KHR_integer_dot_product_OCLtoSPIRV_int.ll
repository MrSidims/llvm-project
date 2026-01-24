; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.6-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.6-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %s -o - -filetype=obj | spirv-val %}

; Test expansion path when native dot product instructions are not available (SPIRV < 1.6, no extension)
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-EXP
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: Capability DotProduct
; CHECK: Capability DotProductInput4x8BitPacked
; CHECK-EXT: OpExtension "SPV_KHR_integer_dot_product"
; CHECK-NOT: OpExtension "SPV_KHR_integer_dot_product"

; CHECK: Name %[[#SignedA:]] "ia"
; CHECK: Name %[[#UnsignedA:]] "ua"
; CHECK: Name %[[#SignedB:]] "ib"
; CHECK: Name %[[#UnsignedB:]] "ub"

; CHECK: SDot %[[#]] %[[#SignedA]] %[[#SignedB]] PackedVectorFormat4x8Bit
; CHECK: SUDot %[[#]] %[[#SignedA]] %[[#UnsignedB]] PackedVectorFormat4x8Bit
; CHECK: SUDot %[[#]] %[[#SignedB]] %[[#UnsignedA]] PackedVectorFormat4x8Bit
; CHECK: UDot %[[#]] %[[#UnsignedA]] %[[#UnsignedB]] PackedVectorFormat4x8Bit

; CHECK: SDotAccSat %[[#]] %[[#SignedA]] %[[#SignedB]] %[[#]] PackedVectorFormat4x8Bit
; CHECK: SUDotAccSat %[[#]] %[[#SignedA]] %[[#UnsignedB]] %[[#]] PackedVectorFormat4x8Bit
; CHECK: SUDotAccSat %[[#]] %[[#SignedB]] %[[#UnsignedA]] %[[#]] PackedVectorFormat4x8Bit
; CHECK: UDotAccSat %[[#]] %[[#UnsignedA]] %[[#UnsignedB]] %[[#]] PackedVectorFormat4x8Bit

; Expansion path checks - verify shift/mask operations for packed 4x8 format
; CHECK-EXP-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0

; Signed packed dot: shift left to sign position, arithmetic shift right to sign-extend
; CHECK-EXP: OpShiftLeftLogical %[[#Int32Ty]]
; CHECK-EXP: OpShiftRightArithmetic %[[#Int32Ty]]
; CHECK-EXP: OpIMul %[[#Int32Ty]]

; Mixed signed/unsigned packed dot: signed operand uses arithmetic shift, unsigned uses logical shift + mask
; CHECK-EXP: OpShiftRightArithmetic %[[#Int32Ty]]
; CHECK-EXP: OpShiftRightLogical %[[#Int32Ty]]
; CHECK-EXP: OpBitwiseAnd %[[#Int32Ty]]
; CHECK-EXP: OpIMul %[[#Int32Ty]]

; Unsigned packed dot: logical shift right and mask with 0xFF
; CHECK-EXP: OpShiftRightLogical %[[#Int32Ty]]
; CHECK-EXP: OpBitwiseAnd %[[#Int32Ty]]
; CHECK-EXP: OpIMul %[[#Int32Ty]]

define spir_kernel void @test(i32 %ia, i32 %ua, i32 %ib, i32 %ub, i32 %ires, i32 %ures, ptr addrspace(1) %out) {
entry:
  ; Store each dot result to prevent DCE during expansion
  %call = tail call spir_func i32 @_Z20dot_4x8packed_ss_intjj(i32 %ia, i32 %ib) #2
  %out0 = getelementptr i32, ptr addrspace(1) %out, i32 0
  store i32 %call, ptr addrspace(1) %out0
  %call1 = tail call spir_func i32 @_Z20dot_4x8packed_su_intjj(i32 %ia, i32 %ub) #2
  %out1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store i32 %call1, ptr addrspace(1) %out1
  %call2 = tail call spir_func i32 @_Z20dot_4x8packed_us_intjj(i32 %ua, i32 %ib) #2
  %out2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  store i32 %call2, ptr addrspace(1) %out2
  %call3 = tail call spir_func i32 @_Z21dot_4x8packed_uu_uintjj(i32 %ua, i32 %ub) #2
  %out3 = getelementptr i32, ptr addrspace(1) %out, i32 3
  store i32 %call3, ptr addrspace(1) %out3
  %call4 = tail call spir_func i32 @_Z28dot_acc_sat_4x8packed_ss_intjji(i32 %ia, i32 %ib, i32 %call2) #2
  %out4 = getelementptr i32, ptr addrspace(1) %out, i32 4
  store i32 %call4, ptr addrspace(1) %out4
  %call5 = tail call spir_func i32 @_Z28dot_acc_sat_4x8packed_su_intjji(i32 %ia, i32 %ub, i32 %call4) #2
  %out5 = getelementptr i32, ptr addrspace(1) %out, i32 5
  store i32 %call5, ptr addrspace(1) %out5
  %call6 = tail call spir_func i32 @_Z28dot_acc_sat_4x8packed_us_intjji(i32 %ua, i32 %ib, i32 %call5) #2
  %out6 = getelementptr i32, ptr addrspace(1) %out, i32 6
  store i32 %call6, ptr addrspace(1) %out6
  %call7 = tail call spir_func i32 @_Z29dot_acc_sat_4x8packed_uu_uintjjj(i32 %ua, i32 %ub, i32 %call6) #2
  %out7 = getelementptr i32, ptr addrspace(1) %out, i32 7
  store i32 %call7, ptr addrspace(1) %out7
  ret void
}

declare spir_func i32 @_Z20dot_4x8packed_ss_intjj(i32, i32)
declare spir_func i32 @_Z20dot_4x8packed_su_intjj(i32, i32)
declare spir_func i32 @_Z20dot_4x8packed_us_intjj(i32, i32)
declare spir_func i32 @_Z21dot_4x8packed_uu_uintjj(i32, i32)
declare spir_func i32 @_Z28dot_acc_sat_4x8packed_ss_intjji(i32, i32, i32)
declare spir_func i32 @_Z28dot_acc_sat_4x8packed_su_intjji(i32, i32, i32)
declare spir_func i32 @_Z28dot_acc_sat_4x8packed_us_intjji(i32, i32, i32)
declare spir_func i32 @_Z29dot_acc_sat_4x8packed_uu_uintjjj(i32, i32, i32)

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
