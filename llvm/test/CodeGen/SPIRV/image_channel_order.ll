; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that get_image_channel_data_type and get_image_channel_order are
; correctly lowered to OpImageQueryFormat and OpImageQueryOrder respectively,
; with the OpenCL-to-SPIR-V enum offset adjustment (IAdd 4304/4272).

; CHECK-DAG: OpCapability ImageBasic
; CHECK-DAG: %[[#INT_TY:]] = OpTypeInt 32
; CHECK-DAG: %[[#DT_OFFSET:]] = OpConstant %[[#INT_TY]] 4304
; CHECK-DAG: %[[#ORD_OFFSET:]] = OpConstant %[[#INT_TY]] 4272

; CHECK: %[[#DT_QUERY:]] = OpImageQueryFormat %[[#INT_TY]]
; CHECK: %[[#DT_ADD:]] = OpIAdd %[[#INT_TY]] %[[#DT_QUERY]] %[[#DT_OFFSET]]
; CHECK: OpStore %[[#]] %[[#DT_ADD]]
; CHECK: %[[#ORD_QUERY:]] = OpImageQueryOrder %[[#INT_TY]]
; CHECK: %[[#ORD_ADD:]] = OpIAdd %[[#INT_TY]] %[[#ORD_QUERY]] %[[#ORD_OFFSET]]
; CHECK: OpStore %[[#]] %[[#ORD_ADD]]

define spir_kernel void @f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img, ptr addrspace(1) %type, ptr addrspace(1) %order) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
  %1 = tail call spir_func i32 @_Z27get_image_channel_data_type14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img)
  store i32 %1, ptr addrspace(1) %type, align 4
  %2 = tail call spir_func i32 @_Z23get_image_channel_order14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img)
  store i32 %2, ptr addrspace(1) %order, align 4
  ret void
}

declare spir_func i32 @_Z27get_image_channel_data_type14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0))
declare spir_func i32 @_Z23get_image_channel_order14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0))

!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}

!0 = !{i32 2, i32 0}
!1 = !{i32 1, i32 1, i32 1}
!2 = !{!"read_only", !"none", !"none"}
!3 = !{!"image2d_t", !"int*", !"int*"}
!4 = !{!"image2d_t", !"int*", !"int*"}
!5 = !{!"", !"", !""}
