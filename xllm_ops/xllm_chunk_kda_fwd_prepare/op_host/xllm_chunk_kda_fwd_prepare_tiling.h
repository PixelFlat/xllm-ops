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

#ifndef XLLM_CHUNK_KDA_FWD_PREPARE_TILING_H
#define XLLM_CHUNK_KDA_FWD_PREPARE_TILING_H

#include <cstddef>

#include "register/tilingdata_base.h"
#include "xllm_chunk_kda_fwd_prepare_output_mask.h"

namespace optiling {

// 字段顺序和类型必须与设备侧读取保持一致。
BEGIN_TILING_DATA_DEF(XllmChunkKdaFwdPrepareTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, seqNum);
TILING_DATA_FIELD_DEF(uint32_t, seqLen);
TILING_DATA_FIELD_DEF(uint32_t, qkHeadNum);
TILING_DATA_FIELD_DEF(uint32_t, valueHeadNum);
TILING_DATA_FIELD_DEF(uint32_t, totalChunks);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, headsPerPartition);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, lowerBound);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(bool, isVarLen);
TILING_DATA_FIELD_DEF(bool, inputSequenceMajor);
TILING_DATA_FIELD_DEF(bool, hasDtBias);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XllmChunkKdaFwdPrepare,
                           XllmChunkKdaFwdPrepareTilingData)

enum XllmChunkKdaFwdPrepareInputIndex : size_t {
  PREPARE_INPUT_Q = 0,
  PREPARE_INPUT_K,
  PREPARE_INPUT_V,
  PREPARE_INPUT_G,
  PREPARE_INPUT_BETA,
  PREPARE_INPUT_A_LOG,
  PREPARE_INPUT_DT_BIAS,
  PREPARE_INPUT_CU_SEQLENS,
  PREPARE_INPUT_CHUNK_INDICES,
};

enum XllmChunkKdaFwdPrepareAttrIndex : size_t {
  PREPARE_ATTR_LAYOUT = 0,
  PREPARE_ATTR_SCALE,
  PREPARE_ATTR_CHUNK_SIZE,
  PREPARE_ATTR_EPSILON,
  PREPARE_ATTR_USE_QK_L2NORM,
  PREPARE_ATTR_USE_GATE,
  PREPARE_ATTR_USE_BETA_SIGMOID,
  PREPARE_ATTR_ALLOW_NEG_EIGVAL,
  PREPARE_ATTR_SAFE_GATE,
  PREPARE_ATTR_LOWER_BOUND,
  PREPARE_ATTR_USE_EXP2,
  PREPARE_ATTR_OUTPUT_MODE,
};

struct XllmChunkKdaFwdPrepareCompileInfo {};

}  // namespace optiling

#endif  // XLLM_CHUNK_KDA_FWD_PREPARE_TILING_H
