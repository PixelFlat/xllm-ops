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

#ifndef OP_API_INC_LEVEL0_XLLM_CHUNK_KDA_FWD_PREPARE_H
#define OP_API_INC_LEVEL0_XLLM_CHUNK_KDA_FWD_PREPARE_H

#include <array>
#include <cstdint>

#include "opdev/op_executor.h"

namespace l0op {

using XllmChunkKdaFwdPrepareOutputs = std::array<const aclTensor*, 13>;

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
    aclOpExecutor* executor);

}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_XLLM_CHUNK_KDA_FWD_PREPARE_H
