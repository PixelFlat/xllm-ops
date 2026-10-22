/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smla_va.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "smla_va_template_tiling_key.h"
#include "arch22/smla_va_csa_kernel.h"
#include "arch22/smla_va_swa_kernel.h"
#include "arch22/smla_va_arch22_metadata.h"

using namespace AscendC;
using namespace optiling::detail;
using namespace SMLAKernel;

#define SMLA_OP_IMPL(templateClass, tilingdataClass, ...) \
    do { \
        templateClass<SMLAType<__VA_ARGS__>> op; \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling); \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in; \
        op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ, \
                cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, seqUsedOriKV, seqUsedCmpKV, cmpResidualKV, oriTopkLength, \
                cmpTopkLength, sinks, metadata, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe); \
        op.Process(); \
    } while (0)

template <int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int SPLIT_G, int HEAD_RATIO_ONE,
          int BATCH_CONSISTENCY, int IS_VEC_S2PHYADDR>
__global__ __aicore__ void smla_va(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV,
    __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    TPipe tPipe;
    if constexpr (ORIG_DTYPE_Q == DT_FLOAT16 && ORIG_DTYPE_ORI_KV == DT_FLOAT16 && ORIG_DTYPE_ATTN_OUT == DT_FLOAT16) {
        if constexpr (TEMPLATE_MODE == CSA_TEMPLATE) {
            SMLA_OP_IMPL(SmlaVaCsa, SmlaVaTilingData, half, half, half, FLASH_DECODE,
                         static_cast<SMLA_LAYOUT>(LAYOUT_T), static_cast<SMLA_LAYOUT>(KV_LAYOUT_T), TEMPLATE_MODE,
                         static_cast<bool>(HEAD_RATIO_ONE));
        } else {
            SMLA_OP_IMPL(SmlaVaSwa, SmlaVaTilingData, half, half, half, FLASH_DECODE,
                         static_cast<SMLA_LAYOUT>(LAYOUT_T), static_cast<SMLA_LAYOUT>(KV_LAYOUT_T), TEMPLATE_MODE,
                         static_cast<bool>(HEAD_RATIO_ONE));
        }
    }
    if constexpr (ORIG_DTYPE_Q == DT_BF16 && ORIG_DTYPE_ORI_KV == DT_BF16 && ORIG_DTYPE_ATTN_OUT == DT_BF16) {
        if constexpr (TEMPLATE_MODE == CSA_TEMPLATE) {
            SMLA_OP_IMPL(SmlaVaCsa, SmlaVaTilingData, bfloat16_t, bfloat16_t, bfloat16_t, FLASH_DECODE,
                         static_cast<SMLA_LAYOUT>(LAYOUT_T), static_cast<SMLA_LAYOUT>(KV_LAYOUT_T), TEMPLATE_MODE,
                         static_cast<bool>(HEAD_RATIO_ONE));
        } else {
            SMLA_OP_IMPL(SmlaVaSwa, SmlaVaTilingData, bfloat16_t, bfloat16_t, bfloat16_t, FLASH_DECODE,
                         static_cast<SMLA_LAYOUT>(LAYOUT_T), static_cast<SMLA_LAYOUT>(KV_LAYOUT_T), TEMPLATE_MODE,
                         static_cast<bool>(HEAD_RATIO_ONE));
        }
    }
}
