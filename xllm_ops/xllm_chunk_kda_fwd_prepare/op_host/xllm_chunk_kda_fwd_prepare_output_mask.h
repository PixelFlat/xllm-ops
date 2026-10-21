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

#ifndef XLLM_CHUNK_KDA_FWD_PREPARE_OUTPUT_MASK_H
#define XLLM_CHUNK_KDA_FWD_PREPARE_OUTPUT_MASK_H

#include <cstddef>
#include <cstdint>

namespace optiling {

enum XllmChunkKdaFwdPrepareOutputMode : int64_t {
  PREPARE_OUTPUT_MODE_NONE = 0,
  PREPARE_OUTPUT_MODE_RECOMPUTE = 1,
  PREPARE_OUTPUT_MODE_SAVE = 2,
  // 只额外搬出 Akk：Aqk/Akk 是公开必选输出，而 qHat/kHat/qRstd/kRstd/betaEff
  // 只服务反向重计算。融合前向链（含 chunk_kda_fwd 的组合入口）用这一档。
  PREPARE_OUTPUT_MODE_FORWARD = 3,
};

// 13 个 IR 输出槽位始终存在。L2 传入 nullptr 时，只关闭对应输出的
// kernel 搬出；outputMask 仅用于 tiling 阶段选择编译期输出模式。
enum XllmChunkKdaFwdPrepareOutputIndex : size_t {
  PREPARE_OUTPUT_GK = 0,
  PREPARE_OUTPUT_AQK,
  PREPARE_OUTPUT_AKK,
  PREPARE_OUTPUT_W,
  PREPARE_OUTPUT_U,
  PREPARE_OUTPUT_QG,
  PREPARE_OUTPUT_KG,
  PREPARE_OUTPUT_QG_SCALED,
  PREPARE_OUTPUT_Q_HAT,
  PREPARE_OUTPUT_K_HAT,
  PREPARE_OUTPUT_Q_RSTD,
  PREPARE_OUTPUT_K_RSTD,
  PREPARE_OUTPUT_BETA_EFF,
  PREPARE_OUTPUT_COUNT,
};

enum XllmChunkKdaFwdPrepareOutputMask : uint32_t {
  PREPARE_OUTPUT_MASK_GK = 1U << PREPARE_OUTPUT_GK,
  PREPARE_OUTPUT_MASK_AQK = 1U << PREPARE_OUTPUT_AQK,
  PREPARE_OUTPUT_MASK_AKK = 1U << PREPARE_OUTPUT_AKK,
  PREPARE_OUTPUT_MASK_W = 1U << PREPARE_OUTPUT_W,
  PREPARE_OUTPUT_MASK_U = 1U << PREPARE_OUTPUT_U,
  PREPARE_OUTPUT_MASK_QG = 1U << PREPARE_OUTPUT_QG,
  PREPARE_OUTPUT_MASK_KG = 1U << PREPARE_OUTPUT_KG,
  PREPARE_OUTPUT_MASK_QG_SCALED = 1U << PREPARE_OUTPUT_QG_SCALED,
  PREPARE_OUTPUT_MASK_Q_HAT = 1U << PREPARE_OUTPUT_Q_HAT,
  PREPARE_OUTPUT_MASK_K_HAT = 1U << PREPARE_OUTPUT_K_HAT,
  PREPARE_OUTPUT_MASK_Q_RSTD = 1U << PREPARE_OUTPUT_Q_RSTD,
  PREPARE_OUTPUT_MASK_K_RSTD = 1U << PREPARE_OUTPUT_K_RSTD,
  PREPARE_OUTPUT_MASK_BETA_EFF = 1U << PREPARE_OUTPUT_BETA_EFF,
};

constexpr uint32_t PREPARE_REQUIRED_OUTPUT_MASK =
    PREPARE_OUTPUT_MASK_GK | PREPARE_OUTPUT_MASK_AQK | PREPARE_OUTPUT_MASK_W |
    PREPARE_OUTPUT_MASK_U | PREPARE_OUTPUT_MASK_KG |
    PREPARE_OUTPUT_MASK_QG_SCALED;

constexpr uint32_t PREPARE_OPTIONAL_OUTPUT_MASK =
    PREPARE_OUTPUT_MASK_AKK | PREPARE_OUTPUT_MASK_QG |
    PREPARE_OUTPUT_MASK_Q_HAT | PREPARE_OUTPUT_MASK_K_HAT |
    PREPARE_OUTPUT_MASK_Q_RSTD | PREPARE_OUTPUT_MASK_K_RSTD |
    PREPARE_OUTPUT_MASK_BETA_EFF;

constexpr uint32_t PREPARE_RECOMPUTE_OUTPUT_MASK =
    PREPARE_REQUIRED_OUTPUT_MASK | PREPARE_OUTPUT_MASK_AKK |
    PREPARE_OUTPUT_MASK_Q_HAT | PREPARE_OUTPUT_MASK_K_HAT |
    PREPARE_OUTPUT_MASK_Q_RSTD | PREPARE_OUTPUT_MASK_K_RSTD |
    PREPARE_OUTPUT_MASK_BETA_EFF;

constexpr uint32_t PREPARE_SAVE_OUTPUT_MASK =
    PREPARE_REQUIRED_OUTPUT_MASK | PREPARE_OPTIONAL_OUTPUT_MASK;

constexpr uint32_t PREPARE_FORWARD_OUTPUT_MASK =
    PREPARE_REQUIRED_OUTPUT_MASK | PREPARE_OUTPUT_MASK_AKK;

static_assert(PREPARE_OUTPUT_COUNT < sizeof(uint32_t) * 8U,
              "outputMask 必须能容纳全部输出位。");
static_assert((PREPARE_REQUIRED_OUTPUT_MASK & PREPARE_OPTIONAL_OUTPUT_MASK) ==
                  0,
              "必选输出和可选输出的掩码不能重叠。");
static_assert((PREPARE_REQUIRED_OUTPUT_MASK | PREPARE_OPTIONAL_OUTPUT_MASK) ==
                  ((1U << PREPARE_OUTPUT_COUNT) - 1U),
              "outputMask 必须覆盖全部输出槽位。");

}  // namespace optiling

#endif  // XLLM_CHUNK_KDA_FWD_PREPARE_OUTPUT_MASK_H
