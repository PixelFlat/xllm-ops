/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ACLNN_MERGE_FWD_BWD_KERNEL_H
#define ACLNN_MERGE_FWD_BWD_KERNEL_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnMergeFwdBwdKernelGetWorkspaceSize(const aclTensor* h,
                                       const aclTensor* agHm,
                                       int64_t preOrPostNumRanks,
                                       int64_t rank,
                                       bool forward,
                                       bool stateVFirst,
                                       uint64_t* workspaceSize,
                                       aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnMergeFwdBwdKernel(void* workspace,
                       uint64_t workspaceSize,
                       aclOpExecutor* executor,
                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
