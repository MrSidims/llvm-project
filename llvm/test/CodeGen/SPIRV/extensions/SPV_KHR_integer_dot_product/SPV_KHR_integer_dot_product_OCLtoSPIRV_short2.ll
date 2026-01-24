; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.6-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.6-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %s -o - -filetype=obj | spirv-val %}

; Test expansion path when native dot product instructions are not available (SPIRV < 1.6, no extension)
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-EXP
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: Capability DotProduct
; CHECK: Capability DotProductInputAll
; CHECK-EXT: OpExtension "SPV_KHR_integer_dot_product"
; CHECK-NOT: OpExtension "SPV_KHR_integer_dot_product"

; CHECK: Name %[[#SignedA:]] "ia"
; CHECK: Name %[[#UnsignedA:]] "ua"
; CHECK: Name %[[#SignedB:]] "ib"
; CHECK: Name %[[#UnsignedB:]] "ub"

; CHECK: SDot %[[#]] %[[#SignedA]] %[[#SignedB]]
; CHECK: SUDot %[[#]] %[[#SignedA]] %[[#UnsignedB]]
; CHECK: SUDot %[[#]] %[[#SignedB]] %[[#UnsignedA]]
; CHECK: UDot %[[#]] %[[#UnsignedA]] %[[#UnsignedB]]

; CHECK: SDotAccSat %[[#]] %[[#SignedA]] %[[#SignedB]] %[[#]]
; CHECK: SUDotAccSat %[[#]] %[[#SignedA]] %[[#UnsignedB]] %[[#]]
; CHECK: SUDotAccSat %[[#]] %[[#SignedB]] %[[#UnsignedA]] %[[#]]
; CHECK: UDotAccSat %[[#]] %[[#UnsignedA]] %[[#UnsignedB]] %[[#]]

; Expansion path checks - verify sign/zero extension and element-wise operations
; CHECK-EXP-DAG: %[[#Int16Ty:]] = OpTypeInt 16 0
; CHECK-EXP-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0

; Signed dot: extract i16, sign-extend to i32, multiply, accumulate
; CHECK-EXP: OpCompositeExtract %[[#Int16Ty]]
; CHECK-EXP: OpSConvert %[[#Int32Ty]]
; CHECK-EXP: OpCompositeExtract %[[#Int16Ty]]
; CHECK-EXP: OpSConvert %[[#Int32Ty]]
; CHECK-EXP: OpIMul %[[#Int32Ty]]

; Mixed signed/unsigned dot: first vector sign-extended, second zero-extended
; CHECK-EXP: OpSConvert %[[#Int32Ty]]
; CHECK-EXP: OpUConvert %[[#Int32Ty]]
; CHECK-EXP: OpIMul %[[#Int32Ty]]

; Unsigned dot: extract i16, zero-extend to i32, multiply, accumulate
; CHECK-EXP: OpUConvert %[[#Int32Ty]]
; CHECK-EXP: OpUConvert %[[#Int32Ty]]
; CHECK-EXP: OpIMul %[[#Int32Ty]]

define spir_kernel void @test(<2 x i16> %ia, <2 x i16> %ua, <2 x i16> %ib, <2 x i16> %ub, <2 x i16> %ires, <2 x i16> %ures, ptr addrspace(1) %out) {
entry:
  ; Store each dot result to prevent DCE during expansion
  %call = tail call spir_func i32 @_Z3dotDv2_sS_(<2 x i16> %ia, <2 x i16> %ib) #2
  %out0 = getelementptr i32, ptr addrspace(1) %out, i32 0
  store i32 %call, ptr addrspace(1) %out0
  %call1 = tail call spir_func i32 @_Z3dotDv2_sDv2_t(<2 x i16> %ia, <2 x i16> %ub) #2
  %out1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store i32 %call1, ptr addrspace(1) %out1
  %call2 = tail call spir_func i32 @_Z3dotDv2_tDv2_s(<2 x i16> %ua, <2 x i16> %ib) #2
  %out2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  store i32 %call2, ptr addrspace(1) %out2
  %call3 = tail call spir_func i32 @_Z3dotDv2_tS_(<2 x i16> %ua, <2 x i16> %ub) #2
  %out3 = getelementptr i32, ptr addrspace(1) %out, i32 3
  store i32 %call3, ptr addrspace(1) %out3
  %call4 = tail call spir_func i32 @_Z11dot_acc_satDv2_sS_i(<2 x i16> %ia, <2 x i16> %ib, i32 %call2) #2
  %out4 = getelementptr i32, ptr addrspace(1) %out, i32 4
  store i32 %call4, ptr addrspace(1) %out4
  %call5 = tail call spir_func i32 @_Z11dot_acc_satDv2_sDv2_ti(<2 x i16> %ia, <2 x i16> %ub, i32 %call4) #2
  %out5 = getelementptr i32, ptr addrspace(1) %out, i32 5
  store i32 %call5, ptr addrspace(1) %out5
  %call6 = tail call spir_func i32 @_Z11dot_acc_satDv2_tDv2_si(<2 x i16> %ua, <2 x i16> %ib, i32 %call5) #2
  %out6 = getelementptr i32, ptr addrspace(1) %out, i32 6
  store i32 %call6, ptr addrspace(1) %out6
  %call7 = tail call spir_func i32 @_Z11dot_acc_satDv2_tS_j(<2 x i16> %ua, <2 x i16> %ub, i32 %call3) #2
  %out7 = getelementptr i32, ptr addrspace(1) %out, i32 7
  store i32 %call7, ptr addrspace(1) %out7
  ret void
}

declare spir_func i32 @_Z3dotDv2_sS_(<2 x i16>, <2 x i16>)
declare spir_func i32 @_Z3dotDv2_sDv2_t(<2 x i16>, <2 x i16>)
declare spir_func i32 @_Z3dotDv2_tDv2_s(<2 x i16>, <2 x i16>)
declare spir_func i32 @_Z3dotDv2_tS_(<2 x i16>, <2 x i16>)
declare spir_func i32 @_Z11dot_acc_satDv2_sS_i(<2 x i16>, <2 x i16>, i32)
declare spir_func i32 @_Z11dot_acc_satDv2_sDv2_ti(<2 x i16>, <2 x i16>, i32)
declare spir_func i32 @_Z11dot_acc_satDv2_tDv2_si(<2 x i16>, <2 x i16>, i32)
declare spir_func i32 @_Z11dot_acc_satDv2_tS_j(<2 x i16>, <2 x i16>, i32)

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
