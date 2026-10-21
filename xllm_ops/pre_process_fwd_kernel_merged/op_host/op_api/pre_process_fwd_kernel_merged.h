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

#ifndef OP_API_INC_LEVEL0_OP_PRE_PROCESS_FWD_KERNEL_MERGED_OP_H
#define OP_API_INC_LEVEL0_OP_PRE_PROCESS_FWD_KERNEL_MERGED_OP_H

#include "opdev/op_executor.h"

namespace l0op {
// 参数顺序与 aclnn / ctypes ARGTYPES 逐参一致；hmOut 为 [Nseq,Hv,K,V+K] FP32。
const std::array<const aclTensor*, 1> PreProcessFwdKernelMerged(
    const aclTensor* k,
    const aclTensor* w,
    const aclTensor* u,
    const aclTensor* g,
    const aclTensor* gk,
    const aclTensor* bg,
    const aclTensor* v,
    const aclIntArray* cuSeqlensOptional,
    int64_t chunkSize,
    const aclTensor* hmOut,
    aclOpExecutor* executor);
}  // namespace l0op

#endif
