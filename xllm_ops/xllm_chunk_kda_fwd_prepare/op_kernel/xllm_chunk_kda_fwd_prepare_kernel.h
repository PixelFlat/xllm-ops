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

#ifndef XLLM_CHUNK_KDA_FWD_PREPARE_KERNEL_H
#define XLLM_CHUNK_KDA_FWD_PREPARE_KERNEL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "xllm_chunk_kda_fwd_prepare_policy.h"
#include "xllm_chunk_kda_fwd_prepare_struct.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/xllm_chunk_kda_fwd_prepare_cube.h"
#include "arch35/xllm_chunk_kda_fwd_prepare_vec.h"
#else
#include "arch22/xllm_chunk_kda_fwd_prepare_cube.h"
#include "arch22/xllm_chunk_kda_fwd_prepare_vec.h"
#endif

namespace KdaPrepare {

template <int DTYPE>
struct PrepareStorageType;

template <>
struct PrepareStorageType<XLLM_CHUNK_KDA_FWD_PREPARE_TPL_BF16> {
  using type = bfloat16_t;
};

template <>
struct PrepareStorageType<XLLM_CHUNK_KDA_FWD_PREPARE_TPL_FP32> {
  using type = float;
};

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void RunPrepare(const PrepareKernelArgs& args) {
  if ASCEND_IS_AIV {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    Arch35::XllmChunkKdaFwdPrepareVec<GateT, BetaT, CompilePolicy> vec;
    vec.Init(args);
#else
    AscendC::TPipe pipe;
    Arch22::XllmChunkKdaFwdPrepareVec<GateT, BetaT, CompilePolicy> vec;
    vec.Init(args, &pipe);
#endif
    vec.Process();
  }

  if ASCEND_IS_AIC {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    Arch35::XllmChunkKdaFwdPrepareCube<CompilePolicy> cube;
    cube.Init(args);
#else
    AscendC::TPipe pipe;
    Arch22::XllmChunkKdaFwdPrepareCube<CompilePolicy> cube;
    cube.Init(args, &pipe);
#endif
    cube.Process();
  }
}

}  // namespace KdaPrepare

#endif  // XLLM_CHUNK_KDA_FWD_PREPARE_KERNEL_H
