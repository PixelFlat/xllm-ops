# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        pre_process_fwd_kernel_merged_def.cpp
    )
endif()

add_modules_sources(OPTYPE pre_process_fwd_kernel_merged ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME PreProcessFwdKernelMerged
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
            -I${OPS_XLLM_COMMON_INC}/kernel
            -I${CANN_3RD_LIB_PATH}/catlass/include
            -I${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include
)
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME PreProcessFwdKernelMerged
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()
