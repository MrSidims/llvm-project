; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that read_imageh (half-precision image read) and write_imageh are
; correctly lowered to SPIR-V ImageRead/ImageSampleExplicitLod/ImageWrite.

; CHECK-DAG: %[[#VOID_TY:]] = OpTypeVoid
; CHECK-DAG: %[[#HALF_TY:]] = OpTypeFloat 16
; CHECK-DAG: %[[#HVEC_TY:]] = OpTypeVector %[[#HALF_TY]] 4
; CHECK-DAG: %[[#IMG_RO:]] = OpTypeImage %[[#VOID_TY]] 2D 0 0 0 0 Unknown ReadOnly
; CHECK-DAG: OpTypeSampler
; CHECK-DAG: %[[#IMG_WO:]] = OpTypeImage %[[#VOID_TY]] 2D 0 0 0 0 Unknown WriteOnly

; CHECK-LABEL: ; -- Begin function nosamp
; CHECK: OpImageRead %[[#HVEC_TY]]
; CHECK: OpFunctionEnd

; CHECK-LABEL: ; -- Begin function withsamp
; CHECK: OpSampledImage
; CHECK: OpImageSampleExplicitLod %[[#HVEC_TY]]
; CHECK: OpFunctionEnd

; CHECK-LABEL: ; -- Begin function writehalf
; CHECK: OpImageWrite
; CHECK: OpFunctionEnd

define spir_kernel void @nosamp(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %im, <2 x i32> %coord, ptr addrspace(1) %res) !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %call = tail call spir_func <4 x half> @_Z11read_imageh14ocl_image2d_roDv2_i(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %im, <2 x i32> %coord)
  store <4 x half> %call, ptr addrspace(1) %res, align 8
  ret void
}

declare spir_func <4 x half> @_Z11read_imageh14ocl_image2d_roDv2_i(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), <2 x i32>)

define spir_kernel void @withsamp(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %im, target("spirv.Sampler") %smp, <2 x i32> %coord, ptr addrspace(1) %res) !kernel_arg_addr_space !11 !kernel_arg_access_qual !12 !kernel_arg_type !13 !kernel_arg_base_type !14 !kernel_arg_type_qual !15 {
entry:
  %call = tail call spir_func <4 x half> @_Z11read_imageh14ocl_image2d_ro11ocl_samplerDv2_i(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %im, target("spirv.Sampler") %smp, <2 x i32> %coord)
  store <4 x half> %call, ptr addrspace(1) %res, align 8
  ret void
}

declare spir_func <4 x half> @_Z11read_imageh14ocl_image2d_ro11ocl_samplerDv2_i(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), <2 x i32>)

define spir_kernel void @writehalf(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %im, <2 x i32> %coord, ptr addrspace(1) readonly %val) !kernel_arg_addr_space !3 !kernel_arg_access_qual !16 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = load <4 x half>, ptr addrspace(1) %val, align 8
  tail call spir_func void @_Z12write_imageh14ocl_image2d_woDv2_iDv4_Dh(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %im, <2 x i32> %coord, <4 x half> %0)
  ret void
}

declare spir_func void @_Z12write_imageh14ocl_image2d_woDv2_iDv4_Dh(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, <4 x half>)

!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!1 = !{i32 2, i32 0}
!3 = !{i32 1, i32 0, i32 1}
!4 = !{!"read_only", !"none", !"none"}
!5 = !{!"image2d_t", !"int2", !"half4*"}
!6 = !{!"image2d_t", !"int __attribute__((ext_vector_type(2)))", !"half __attribute__((ext_vector_type(4)))*"}
!7 = !{!"", !"", !""}
!11 = !{i32 1, i32 0, i32 0, i32 1}
!12 = !{!"read_only", !"none", !"none", !"none"}
!13 = !{!"image2d_t", !"sampler_t", !"int2", !"half4*"}
!14 = !{!"image2d_t", !"sampler_t", !"int __attribute__((ext_vector_type(2)))", !"half __attribute__((ext_vector_type(4)))*"}
!15 = !{!"", !"", !"", !""}
!16 = !{!"write_only", !"none", !"none"}
