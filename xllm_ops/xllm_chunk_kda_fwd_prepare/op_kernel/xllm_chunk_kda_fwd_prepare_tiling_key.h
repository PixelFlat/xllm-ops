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

#ifndef XLLM_CHUNK_KDA_FWD_PREPARE_TILING_KEY_H
#define XLLM_CHUNK_KDA_FWD_PREPARE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "xllm_chunk_kda_fwd_prepare_policy.h"

namespace KdaPrepare {

ASCENDC_TPL_ARGS_DECL(
    XllmChunkKdaFwdPrepare,
    ASCENDC_TPL_DTYPE_DECL(D_T_GATE,
                           XLLM_CHUNK_KDA_FWD_PREPARE_TPL_BF16,
                           XLLM_CHUNK_KDA_FWD_PREPARE_TPL_FP32),
    ASCENDC_TPL_DTYPE_DECL(D_T_BETA,
                           XLLM_CHUNK_KDA_FWD_PREPARE_TPL_BF16,
                           XLLM_CHUNK_KDA_FWD_PREPARE_TPL_FP32),
    ASCENDC_TPL_UINT_DECL(NORM_MODE,
                          1,
                          ASCENDC_TPL_UI_LIST,
                          XLLM_CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY,
                          XLLM_CHUNK_KDA_FWD_PREPARE_NORM_L2),
    ASCENDC_TPL_UINT_DECL(BETA_MODE,
                          2,
                          ASCENDC_TPL_UI_LIST,
                          XLLM_CHUNK_KDA_FWD_PREPARE_BETA_RAW,
                          XLLM_CHUNK_KDA_FWD_PREPARE_BETA_SIGMOID,
                          XLLM_CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID),
    ASCENDC_TPL_UINT_DECL(GATE_MODE,
                          2,
                          ASCENDC_TPL_UI_LIST,
                          XLLM_CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP,
                          XLLM_CHUNK_KDA_FWD_PREPARE_GATE_SOFTPLUS,
                          XLLM_CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID),
    ASCENDC_TPL_BOOL_DECL(USE_EXP2, 0, 1),
    ASCENDC_TPL_BOOL_DECL(SAFE_GATE, 0, 1),
    ASCENDC_TPL_UINT_DECL(OUTPUT_MODE,
                          2,
                          ASCENDC_TPL_UI_LIST,
                          XLLM_CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE,
                          XLLM_CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE,
                          XLLM_CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE,
                          XLLM_CHUNK_KDA_FWD_PREPARE_OUTPUT_FORWARD));

// SAFE_GATE 只为 SafeSigmoid 置位，避免生成数学等价的重复实例。
#define XLLM_CHUNK_KDA_FWD_PREPARE_SEL_ONE(GATE_TYPE,                   \
                                           BETA_TYPE,                   \
                                           NORM_VALUE,                  \
                                           BETA_VALUE,                  \
                                           GATE_VALUE,                  \
                                           EXP_VALUE,                   \
                                           SAFE_VALUE,                  \
                                           OUTPUT_VALUE)                \
  ASCENDC_TPL_ARGS_SEL(                                                 \
      ASCENDC_TPL_DTYPE_SEL(D_T_GATE, GATE_TYPE),                       \
      ASCENDC_TPL_DTYPE_SEL(D_T_BETA, BETA_TYPE),                       \
      ASCENDC_TPL_UINT_SEL(NORM_MODE, ASCENDC_TPL_UI_LIST, NORM_VALUE), \
      ASCENDC_TPL_UINT_SEL(BETA_MODE, ASCENDC_TPL_UI_LIST, BETA_VALUE), \
      ASCENDC_TPL_UINT_SEL(GATE_MODE, ASCENDC_TPL_UI_LIST, GATE_VALUE), \
      ASCENDC_TPL_BOOL_SEL(USE_EXP2, EXP_VALUE),                        \
      ASCENDC_TPL_BOOL_SEL(SAFE_GATE, SAFE_VALUE),                      \
      ASCENDC_TPL_UINT_SEL(OUTPUT_MODE, ASCENDC_TPL_UI_LIST, OUTPUT_VALUE))

#define XLLM_CHUNK_KDA_FWD_PREPARE_SEL_OUTPUT(GATE_TYPE,                      \
                                              BETA_TYPE,                      \
                                              NORM_VALUE,                     \
                                              BETA_VALUE,                     \
                                              GATE_VALUE,                     \
                                              EXP_VALUE,                      \
                                              SAFE_VALUE)                     \
  XLLM_CHUNK_KDA_FWD_PREPARE_SEL_ONE(GATE_TYPE,                               \
                                     BETA_TYPE,                               \
                                     NORM_VALUE,                              \
                                     BETA_VALUE,                              \
                                     GATE_VALUE,                              \
                                     EXP_VALUE,                               \
                                     SAFE_VALUE,                              \
                                     XLLM_CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE), \
      XLLM_CHUNK_KDA_FWD_PREPARE_SEL_ONE(                                     \
          GATE_TYPE,                                                          \
          BETA_TYPE,                                                          \
          NORM_VALUE,                                                         \
          BETA_VALUE,                                                         \
          GATE_VALUE,                                                         \
          EXP_VALUE,                                                          \
          SAFE_VALUE,                                                         \
          XLLM_CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE),                       \
      XLLM_CHUNK_KDA_FWD_PREPARE_SEL_ONE(                                     \
          GATE_TYPE,                                                          \
          BETA_TYPE,                                                          \
          NORM_VALUE,                                                         \
          BETA_VALUE,                                                         \
          GATE_VALUE,                                                         \
          EXP_VALUE,                                                          \
          SAFE_VALUE,                                                         \
          XLLM_CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE),                            \
      XLLM_CHUNK_KDA_FWD_PREPARE_SEL_ONE(                                     \
          GATE_TYPE,                                                          \
          BETA_TYPE,                                                          \
          NORM_VALUE,                                                         \
          BETA_VALUE,                                                         \
          GATE_VALUE,                                                         \
          EXP_VALUE,                                                          \
          SAFE_VALUE,                                                         \
          XLLM_CHUNK_KDA_FWD_PREPARE_OUTPUT_FORWARD)

#define XLLM_CHUNK_KDA_FWD_PREPARE_SEL_EXP(                               \
    GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE, SAFE_VALUE) \
  XLLM_CHUNK_KDA_FWD_PREPARE_SEL_OUTPUT(GATE_TYPE,                        \
                                        BETA_TYPE,                        \
                                        NORM_VALUE,                       \
                                        BETA_VALUE,                       \
                                        GATE_VALUE,                       \
                                        0,                                \
                                        SAFE_VALUE),                      \
      XLLM_CHUNK_KDA_FWD_PREPARE_SEL_OUTPUT(GATE_TYPE,                    \
                                            BETA_TYPE,                    \
                                            NORM_VALUE,                   \
                                            BETA_VALUE,                   \
                                            GATE_VALUE,                   \
                                            1,                            \
                                            SAFE_VALUE)

#define XLLM_CHUNK_KDA_FWD_PREPARE_SEL_GATE(            \
    GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE)       \
  XLLM_CHUNK_KDA_FWD_PREPARE_SEL_EXP(                   \
      GATE_TYPE,                                        \
      BETA_TYPE,                                        \
      NORM_VALUE,                                       \
      BETA_VALUE,                                       \
      XLLM_CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP, \
      0),                                               \
      XLLM_CHUNK_KDA_FWD_PREPARE_SEL_EXP(               \
          GATE_TYPE,                                    \
          BETA_TYPE,                                    \
          NORM_VALUE,                                   \
          BETA_VALUE,                                   \
          XLLM_CHUNK_KDA_FWD_PREPARE_GATE_SOFTPLUS,     \
          0),                                           \
      XLLM_CHUNK_KDA_FWD_PREPARE_SEL_EXP(               \
          GATE_TYPE,                                    \
          BETA_TYPE,                                    \
          NORM_VALUE,                                   \
          BETA_VALUE,                                   \
          XLLM_CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID, \
          1)

#define XLLM_CHUNK_KDA_FWD_PREPARE_SEL_BETA_MODE(                             \
    GATE_TYPE, BETA_TYPE, NORM_VALUE)                                         \
  XLLM_CHUNK_KDA_FWD_PREPARE_SEL_GATE(                                        \
      GATE_TYPE, BETA_TYPE, NORM_VALUE, XLLM_CHUNK_KDA_FWD_PREPARE_BETA_RAW), \
      XLLM_CHUNK_KDA_FWD_PREPARE_SEL_GATE(                                    \
          GATE_TYPE,                                                          \
          BETA_TYPE,                                                          \
          NORM_VALUE,                                                         \
          XLLM_CHUNK_KDA_FWD_PREPARE_BETA_SIGMOID),                           \
      XLLM_CHUNK_KDA_FWD_PREPARE_SEL_GATE(                                    \
          GATE_TYPE,                                                          \
          BETA_TYPE,                                                          \
          NORM_VALUE,                                                         \
          XLLM_CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID)

#define XLLM_CHUNK_KDA_FWD_PREPARE_SEL_NORM(GATE_TYPE, BETA_TYPE)      \
  XLLM_CHUNK_KDA_FWD_PREPARE_SEL_BETA_MODE(                            \
      GATE_TYPE, BETA_TYPE, XLLM_CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY), \
      XLLM_CHUNK_KDA_FWD_PREPARE_SEL_BETA_MODE(                        \
          GATE_TYPE, BETA_TYPE, XLLM_CHUNK_KDA_FWD_PREPARE_NORM_L2)

#define XLLM_CHUNK_KDA_FWD_PREPARE_SEL_BETA_TYPE(GATE_TYPE)                 \
  XLLM_CHUNK_KDA_FWD_PREPARE_SEL_NORM(GATE_TYPE,                            \
                                      XLLM_CHUNK_KDA_FWD_PREPARE_TPL_BF16), \
      XLLM_CHUNK_KDA_FWD_PREPARE_SEL_NORM(GATE_TYPE,                        \
                                          XLLM_CHUNK_KDA_FWD_PREPARE_TPL_FP32)

ASCENDC_TPL_SEL(XLLM_CHUNK_KDA_FWD_PREPARE_SEL_BETA_TYPE(
                    XLLM_CHUNK_KDA_FWD_PREPARE_TPL_BF16),
                XLLM_CHUNK_KDA_FWD_PREPARE_SEL_BETA_TYPE(
                    XLLM_CHUNK_KDA_FWD_PREPARE_TPL_FP32));

#undef XLLM_CHUNK_KDA_FWD_PREPARE_SEL_BETA_TYPE
#undef XLLM_CHUNK_KDA_FWD_PREPARE_SEL_NORM
#undef XLLM_CHUNK_KDA_FWD_PREPARE_SEL_BETA_MODE
#undef XLLM_CHUNK_KDA_FWD_PREPARE_SEL_GATE
#undef XLLM_CHUNK_KDA_FWD_PREPARE_SEL_EXP
#undef XLLM_CHUNK_KDA_FWD_PREPARE_SEL_OUTPUT
#undef XLLM_CHUNK_KDA_FWD_PREPARE_SEL_ONE

}  // namespace KdaPrepare

#endif  // XLLM_CHUNK_KDA_FWD_PREPARE_TILING_KEY_H
