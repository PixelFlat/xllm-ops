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

#include "pre_process_fwd_kernel_merged.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(PreProcessFwdKernelMerged);

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
    aclOpExecutor* executor) {
  L0_DFX(PreProcessFwdKernelMerged,
         k,
         w,
         u,
         g,
         gk,
         bg,
         v,
         cuSeqlensOptional,
         chunkSize,
         hmOut);

  // cu_seqlens 是 host int 数组：转成 INT64 tensor 后作为第 8 个输入传给 kernel
  const aclTensor* actualCuSeqlens = nullptr;
  if (cuSeqlensOptional != nullptr) {
    actualCuSeqlens =
        executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
    const_cast<aclTensor*>(actualCuSeqlens)
        ->SetStorageFormat(Format::FORMAT_ND);
    const_cast<aclTensor*>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
    const_cast<aclTensor*>(actualCuSeqlens)
        ->SetOriginalFormat(Format::FORMAT_ND);
  }

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
      PreProcessFwdKernelMerged,
      OP_INPUT(k, w, u, g, gk, bg, v, actualCuSeqlens),
      OP_OUTPUT(hmOut),
      OP_ATTR(chunkSize));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return {nullptr};
  }
  return {hmOut};
}

}  // namespace l0op
