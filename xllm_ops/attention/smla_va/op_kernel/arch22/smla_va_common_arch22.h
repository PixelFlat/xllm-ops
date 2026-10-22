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
 * \file smla_va_common_arch22.h
 * \brief
 */

#ifndef SMLA_VA_COMMON_ARCH22_H
#define SMLA_VA_COMMON_ARCH22_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

namespace SMLAKernel {
using namespace AscendC;
// When isCheckTiling is false, max, sum, and exp inputs/outputs have shape (m, 1).
constexpr SoftmaxConfig SMLA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

enum class SMLA_RUN_MODE {
    SWA_MODE = 0,
    CSA_MODE = 1,
    HCA_MODE = 2,
};

enum class SMLA_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

template <typename Q_T, typename KV_T, typename OUT_T, const bool FLASH_DECODE = false,
          SMLA_LAYOUT LAYOUT_T = SMLA_LAYOUT::BSND, SMLA_LAYOUT KV_LAYOUT_T = SMLA_LAYOUT::PA_BBND,
          int TEMPLATE_MODE = 0, const bool HEAD_RATIO_ONE = false, typename... Args>
struct SMLAType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr SMLA_LAYOUT layout = LAYOUT_T;
    static constexpr SMLA_LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr bool pageAttention = (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND);
    static constexpr int templateMode = TEMPLATE_MODE;
    static constexpr bool headRatioOne = HEAD_RATIO_ONE;
};

// ================================Util functions==================================
template <typename T1, typename T2>
__aicore__ inline T1 SMLAAlign(T1 num, T2 rnd)
{
    return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd * rnd);
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 num, T2 rnd)
{
    return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? b : a;
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? a : b;
}

template <typename T>
__aicore__ inline size_t BlockAlign(size_t s)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return (s + 63) / 64 * 64;
    }
    size_t n = (32 / sizeof(T));
    return (s + n - 1) / n * n;
}

struct PAShape {
    uint32_t blockSize;
    uint32_t headNum; // Usually the KV head count, corresponding to N2.
    uint32_t headDim; // 512 corresponds to D.
    uint32_t kvStride;
    uint32_t maxblockNumPerBatch; // Maximum number of entries in each block-table row.
    uint32_t actHeadDim;          // Actual number of columns to copy, accounting for N tiling; S*D corresponds to D.
    uint32_t copyRowNum;          // Total number of rows to copy.
    uint32_t copyRowNumAlign;
};

struct Position {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Idx;
    uint32_t dIdx;
};

// Scenario: copy query, key, and value from GM to L1.
// GM uses ND layout; L1 uses NZ layout.
// GM row count, column count, and column stride.
template <typename T>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<T> &l1Tensor, GlobalTensor<T> &gmTensor, uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col,       // D
                                        uint32_t colStride) // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = rowAct; // Number of rows in the ND matrix.
    // For int4 T, dValue = col / 2 and srcDValue = colStride / 2.
    nd2nzPara.dValue = col;          // Number of columns in the ND matrix.
    nd2nzPara.srcDValue = colStride; // Offset between the start addresses of adjacent ND rows.
    nd2nzPara.dstNzC0Stride = rowAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

/*
    Copy PA data from GM to L1; ND and NZ layouts are supported.
    PA layouts include BBND (blockNum, N, blockSize, D) and BBH (blockNum, blockSize, N*D).
    BSH, BSND, and TND use BBH.
    shape.copyRowNumAlign must be aligned to 16 bytes. For example, when copying a K matrix in a 128*512 tile, a 10*512 tail is aligned to 16*512.
*/
template <typename T, SMLA_LAYOUT SRC_LAYOUT = SMLA_LAYOUT::PA_BBND>
__aicore__ inline void DataCopyPA(LocalTensor<T> &dstTensor,  // l1
                                  GlobalTensor<T> &srcTensor, // gm
                                  GlobalTensor<int32_t> &blockTableGm,
                                  const PAShape &shape,     // blockSize, headNum, headDim
                                  const Position &startPos) // bacthIdx nIdx curSeqIdx
{
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
    uint32_t curS2Idx = startPos.s2Idx;
    uint32_t blockElementCnt = 32 / sizeof(T);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize; // Index into the block table.
        uint64_t reaminRowCnt = curS2Idx % shape.blockSize;  // Rows past the start of the current block.
        uint64_t idInBlockTable =
            blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // Get the block ID from the block table.
        uint32_t copyRowCnt = shape.blockSize - reaminRowCnt;            // Process one block at a time.
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt; // The block is not fully copied.
        }
        uint64_t offset = idInBlockTable * shape.kvStride; // Offset in PA storage.
        uint64_t dStride = shape.headDim;
        if constexpr (SRC_LAYOUT == SMLA_LAYOUT::BSND || SRC_LAYOUT == SMLA_LAYOUT::TND) {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim) + reaminRowCnt * shape.headDim * shape.headNum +
                      startPos.dIdx;
            dStride = shape.headDim * shape.headNum;
        } else {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim * shape.blockSize) + reaminRowCnt * shape.headDim +
                      startPos.dIdx;
        }

        uint32_t dValue = shape.actHeadDim;
        uint32_t srcDValue = dStride;
        LocalTensor<T> tmpDstTensor = dstTensor[copyFinishRowCnt * blockElementCnt];
        GlobalTensor<T> tmpSrcTensor = srcTensor[offset];

        DataCopyGmNDToL1<T>(tmpDstTensor, tmpSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template <typename T>
__aicore__ inline void DataCopyPABySlots(LocalTensor<T> &dstTensor, GlobalTensor<T> &srcTensor,
                                         GlobalTensor<int32_t> &blockTableGm, GlobalTensor<int32_t> &sparseIndicesGm,
                                         const PAShape &shape, const Position &startPos, uint64_t sparseIndexBaseOffset,
                                         uint32_t sparseIndexStart)
{
    uint32_t blockElementCnt = 32 / sizeof(T);
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
    for (uint32_t row = 0; row < shape.copyRowNum; ++row) {
        int32_t logicalIdx = sparseIndicesGm.GetValue(sparseIndexBaseOffset + sparseIndexStart + row);
        if (logicalIdx < 0) {
            continue;
        }
        uint64_t blockTableIdx = static_cast<uint64_t>(logicalIdx) / shape.blockSize;
        uint64_t inBlockIdx = static_cast<uint64_t>(logicalIdx) % shape.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockTableIdx);
        uint64_t offset = idInBlockTable * shape.kvStride;
        offset += static_cast<uint64_t>(startPos.n2Idx * shape.headDim * shape.blockSize) + inBlockIdx * shape.headDim +
                  startPos.dIdx;

        LocalTensor<T> tmpDstTensor = dstTensor[row * blockElementCnt];
        GlobalTensor<T> tmpSrcTensor = srcTensor[offset];
        DataCopyGmNDToL1<T>(tmpDstTensor, tmpSrcTensor, 1, shape.copyRowNumAlign, shape.actHeadDim, shape.headDim);
    }
}

struct RunInfo {
    uint32_t loop = 0;
    uint32_t cmpLoop = 0; // Selects which of the four GM blocks used for merging to read.
    uint32_t bIdx = 0;
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t n2IdxReal = 0;
    uint32_t relativeS2Idx = 0;
    uint32_t bn2IdxInCurCore = 0;
    uint32_t curSInnerLoopTimes = 0;
    uint64_t tndBIdxOffsetForQ = 0;
    uint64_t tndBIdxOffsetForKV = 0;
    uint64_t tensorCmpBOffset = 0;
    uint64_t tensorAOffset = 0;
    uint64_t tensorBOffset = 0;
    uint64_t attenOutOffset = 0;
    uint64_t qTokenOffset = 0;
    uint64_t attenMaskOffset = 0;
    uint64_t topKBaseOffset = 0;
    uint32_t actualSingleProcessSInnerSize = 0;
    uint32_t actualSingleProcessSInnerSizeAlign = 0;
    uint32_t actualSingleProcessSInnerOriSize = 0;
    uint32_t actualSingleProcessSInnerOriAlignSize = 0;
    uint32_t actualSingleProcessSInnerCmpSize = 0;
    uint32_t actualSingleProcessSInnerCmpAlignSize = 0;
    uint32_t s2BatchOffset = 0;
    uint32_t gSize = 0;
    uint32_t s1Size = 0;
    uint32_t s2Size = 0;
    uint32_t mSize = 0;
    uint32_t mSizeV = 0;
    uint32_t mSizeVStart = 0;
    uint32_t tndIsS2SplitCore = 0;
    uint32_t tndCoreStartKVSplitPos = 0;
    bool isFirstSInnerLoop = false;
    bool isBmm2Output = false;
    bool isValid = false;
    bool isLastS2Loop = 0;
    int64_t inValidRowCount = 0;

    uint64_t actS1Size = 1;
    uint64_t actS2SizeOri = 0ULL;
    static constexpr uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint64_t actS2Size = 1;
    uint64_t actOriS2Size = 1;
    uint32_t actMBaseSize = 0;
    int32_t nextTokensPerBatch = 0;
    int64_t threshold = 0;
    uint64_t curOffsetInSparseBlock = 0;
    uint32_t curTopKIdx = 0;
    bool isOriOnly = true; // Whether the current block belongs to the original or compressed part.
    bool isOriCmpMix = false;
    uint8_t resv[2];
    uint64_t s2StartPoint = 0;
    int64_t cmpS2IdStart = 0;
    int64_t cmpS2IdLimit = 0;
    int32_t v0S2DealSize = 0;
    int32_t v0S2Start = 0;
    uint32_t oriDealSize = 0;
    int32_t cmpMaskRight = 0;
};

struct ConstInfo {
    // Synchronization mode between CUBE and VEC cores.
    static constexpr uint32_t SMLA_SYNC_MODE2 = 2;
    // Buffer size in bytes.
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    // FP32 zero and maximum finite values.
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float FLOAT_MAX = 3.402823466e+38F;

    // Total number of preloads.
    uint32_t preLoadNum = 0U;
    uint32_t nBufferMBaseSize = 0U;
    // Inter-core synchronization event IDs for CUBE and VEC.
    uint32_t syncV0C1 = 0U;
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C2 = 0U;
    uint32_t syncC2V2 = 0U;

    uint32_t mmResUbSize = 0U;   // GM size for Matmul1 output.
    uint32_t vec1ResUbSize = 0U; // GM size for Vector1 output.
    uint32_t bmm2ResUbSize = 0U; // GM size for Matmul2 output.
    uint32_t usedCoreNum = 0U;
    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0;
    uint64_t headDim = 0;
    uint64_t kvSeqSize = 0ULL;    // Maximum KV sequence length.
    uint64_t qSeqSize = 1ULL;     // Maximum query sequence length.
    int64_t kvCacheBlockSize = 0; // Block size for PA.
    uint64_t paCmpBlockSize = 0;
    uint64_t paOriBlockSize = 0;
    int64_t orikvCacheBlockSize = 0;
    int64_t cmpkvCacheBlockSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0; // Maximum number of blocks per batch for PA.
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t splitKVNum = 0U; // Number of inter-core splits along S2.
    SMLA_LAYOUT outputLayout; // Output transpose layout.
    uint32_t oriMaskMode = 0;
    uint32_t cmpMaskMode = 0;
    uint64_t oriKvStride0 = 0;
    uint64_t cmpKvStride0 = 0;
    bool needInit = false;
    uint32_t templateMode = 0;

    // FlashDecoding
    uint32_t actualCombineLoopSize = 0U; // Maximum number of inter-core S2 splits for FlashDecoding.
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;

    uint32_t actualLenDimsQ = 0U;  // Dimension of query actualSeqLength.
    uint32_t actualLenDimsKV = 0U; // Dimension of KV actualSeqLength.

    uint32_t actualLenDimsCmpKV = 0U;
    uint32_t cmpResidualKVSize = 0U;

    // TND
    uint32_t s2Start = 0U; // S2 start position for TND.
    uint32_t s2End = 0U;   // S2 loop index limit for single-core TND.

    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;

    uint32_t tndFDCoreArrLen = 0U;     // Length of the TND FlashDecoding core-split metadata array.
    uint32_t coreStartKVSplitPos = 0U; // KV start position for TND FlashDecoding.

    uint32_t mBaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    // sparse attr
    int64_t sparseBlockSize = 0;
    uint32_t sparseBlockCount = 0;
    bool hasOriSparseIndices = false;
    bool hasOriTopkLength = false;
    uint32_t oriSparseIndexWidth = 0;

    // cmp attr
    int64_t cmpRatio = 0;

    uint64_t cmpSeqSize = 0ULL;

    // win
    int32_t oriWinRight = 0;
    int32_t oriWinLeft = 128;

    bool returnSoftmaxLse = false;
};

struct MSplitInfo {
    uint32_t nBufferIdx = 0U;
    uint32_t nBufferStartM = 0U;
    uint32_t nBufferDealM = 0U;
    uint32_t vecStartM = 0U;
    uint32_t vecDealM = 0U;
};
} // namespace SMLAKernel
#endif // SMLA_VA_COMMON_ARCH22_H
