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

#include "xllm_chunk_kda_fwd_prepare.h"

#include <cstdint>
#include <initializer_list>

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(XllmChunkKdaFwdPrepare);

namespace {

op::Shape MakeShape(std::initializer_list<int64_t> dims) {
  op::Shape shape;
  for (int64_t dim : dims) {
    shape.AppendDim(dim);
  }
  return shape;
}

const aclTensor* OutputOrEmptyDescriptor(const aclTensor* output,
                                         DataType dtype,
                                         aclOpExecutor* executor) {
  if (output != nullptr) {
    return output;
  }
  return executor->AllocTensor(MakeShape({0}), dtype, Format::FORMAT_ND);
}

const aclTensor* ConvertIntArrayToTensor(const aclIntArray* array,
                                         aclOpExecutor* executor) {
  if (array == nullptr) {
    return nullptr;
  }
  const aclTensor* tensor =
      executor->ConvertToTensor(array, DataType::DT_INT64);
  if (tensor == nullptr) {
    return nullptr;
  }
  auto* mutableTensor = const_cast<aclTensor*>(tensor);
  mutableTensor->SetStorageFormat(Format::FORMAT_ND);
  mutableTensor->SetViewFormat(Format::FORMAT_ND);
  mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
  return tensor;
}

}  // namespace

XllmChunkKdaFwdPrepareOutputs XllmChunkKdaFwdPrepare(
    const aclTensor* q,
    const aclTensor* k,
    const aclTensor* v,
    const aclTensor* g,
    const aclTensor* beta,
    const aclTensor* aLogOptional,
    const aclTensor* dtBiasOptional,
    const aclIntArray* cuSeqlensOptional,
    const aclIntArray* chunkIndicesOptional,
    const char* layout,
    double scale,
    int64_t chunkSize,
    double epsilon,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool useExp2,
    const aclTensor* gkOut,
    const aclTensor* aqkOut,
    const aclTensor* akkOut,
    const aclTensor* wOut,
    const aclTensor* uOut,
    const aclTensor* qgOut,
    const aclTensor* kgOut,
    const aclTensor* qgScaledOut,
    const aclTensor* qHatOut,
    const aclTensor* kHatOut,
    const aclTensor* qRstdOut,
    const aclTensor* kRstdOut,
    const aclTensor* betaEffOut,
    int64_t outputMode,
    aclOpExecutor* executor) {
  L0_DFX(XllmChunkKdaFwdPrepare,
         q,
         k,
         v,
         g,
         beta,
         aLogOptional,
         dtBiasOptional,
         cuSeqlensOptional,
         chunkIndicesOptional,
         layout,
         scale,
         chunkSize,
         epsilon,
         useQkL2normInKernel,
         useGateInKernel,
         useBetaSigmoidInKernel,
         allowNegEigval,
         safeGate,
         lowerBound,
         useExp2,
         gkOut,
         aqkOut,
         akkOut,
         wOut,
         uOut,
         qgOut,
         kgOut,
         qgScaledOut,
         qHatOut,
         kHatOut,
         qRstdOut,
         kRstdOut,
         betaEffOut,
         outputMode);

  const aclTensor* cuSeqlens =
      ConvertIntArrayToTensor(cuSeqlensOptional, executor);
  const aclTensor* chunkIndices =
      ConvertIntArrayToTensor(chunkIndicesOptional, executor);
  if ((cuSeqlensOptional != nullptr && cuSeqlens == nullptr) ||
      (chunkIndicesOptional != nullptr && chunkIndices == nullptr)) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
            "转换 cu_seqlens/chunk_indices 到 Tensor 失败。");
    return {};
  }

  const float scaleAttr = static_cast<float>(scale);
  const float epsilonAttr = static_cast<float>(epsilon);
  const float lowerBoundAttr = static_cast<float>(lowerBound);

  // REQUIRED 输出不能把 nullptr 直接交给 launcher，否则部分 CANN 版本会
  // 压缩参数并使后续 kernel 参数错位。零元素 descriptor 不承载真实数据，
  // 编译期 outputMode 同时保证对应 kernel 形参不发生搬出。
  const aclTensor* akkForKernel =
      OutputOrEmptyDescriptor(akkOut, DataType::DT_BF16, executor);
  const aclTensor* qgForKernel =
      OutputOrEmptyDescriptor(qgOut, DataType::DT_BF16, executor);
  const aclTensor* qHatForKernel =
      OutputOrEmptyDescriptor(qHatOut, DataType::DT_BF16, executor);
  const aclTensor* kHatForKernel =
      OutputOrEmptyDescriptor(kHatOut, DataType::DT_BF16, executor);
  const aclTensor* qRstdForKernel =
      OutputOrEmptyDescriptor(qRstdOut, DataType::DT_FLOAT, executor);
  const aclTensor* kRstdForKernel =
      OutputOrEmptyDescriptor(kRstdOut, DataType::DT_FLOAT, executor);
  const aclTensor* betaEffForKernel =
      OutputOrEmptyDescriptor(betaEffOut, DataType::DT_FLOAT, executor);
  if (akkForKernel == nullptr || qgForKernel == nullptr ||
      qHatForKernel == nullptr || kHatForKernel == nullptr ||
      qRstdForKernel == nullptr || kRstdForKernel == nullptr ||
      betaEffForKernel == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "创建空输出 descriptor 失败。");
    return {};
  }

  const auto status =
      ADD_TO_LAUNCHER_LIST_AICORE(XllmChunkKdaFwdPrepare,
                                  OP_INPUT(q,
                                           k,
                                           v,
                                           g,
                                           beta,
                                           aLogOptional,
                                           dtBiasOptional,
                                           cuSeqlens,
                                           chunkIndices),
                                  OP_OUTPUT(gkOut,
                                            aqkOut,
                                            akkForKernel,
                                            wOut,
                                            uOut,
                                            qgForKernel,
                                            kgOut,
                                            qgScaledOut,
                                            qHatForKernel,
                                            kHatForKernel,
                                            qRstdForKernel,
                                            kRstdForKernel,
                                            betaEffForKernel),
                                  OP_ATTR(layout,
                                          scaleAttr,
                                          chunkSize,
                                          epsilonAttr,
                                          useQkL2normInKernel,
                                          useGateInKernel,
                                          useBetaSigmoidInKernel,
                                          allowNegEigval,
                                          safeGate,
                                          lowerBoundAttr,
                                          useExp2,
                                          outputMode));
  if (status != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "添加 XllmChunkKdaFwdPrepare AI Core 任务失败。");
    return {};
  }

  return {gkOut,
          aqkOut,
          akkOut,
          wOut,
          uOut,
          qgOut,
          kgOut,
          qgScaledOut,
          qHatOut,
          kHatOut,
          qRstdOut,
          kRstdOut,
          betaEffOut};
}

}  // namespace l0op
