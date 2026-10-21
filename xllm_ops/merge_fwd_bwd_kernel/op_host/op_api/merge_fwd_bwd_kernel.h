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

#ifndef OP_API_INC_LEVEL0_OP_MERGE_FWD_BWD_KERNEL_H
#define OP_API_INC_LEVEL0_OP_MERGE_FWD_BWD_KERNEL_H

#include <array>

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor*, 1> MergeFwdBwdKernel(
    const aclTensor* h,
    const aclTensor* agHm,
    int64_t preOrPostNumRanks,
    int64_t rank,
    bool forward,
    bool stateVFirst,
    aclOpExecutor* executor);
}  // namespace l0op

#endif
