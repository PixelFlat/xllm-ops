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
 * \file smla_va_csa_block_vector.h
 * \brief
 */
#ifndef SMLA_VA_CSA_BLOCK_VECTOR_H
#define SMLA_VA_CSA_BLOCK_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "smla_va_common_arch22.h"

namespace SMLAKernel {
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SMLAT>
class SMLAVectorBlock {
public:
    // Use float for intermediate calculations in high-precision mode.
    using T = float;
    using KV_T = typename SMLAT::kvType;
    using OUT_T = typename SMLAT::outputType;
    using UPDATE_T = T;
    using SINKS_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline SMLAVectorBlock(){};
    __aicore__ inline void ProcessVec0L(const RunInfo &runInfo);
    __aicore__ inline void ProcessVec1L(const RunInfo &info);
    __aicore__ inline void ProcessVec2L(const RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo,
                                      const SmlaVaTilingData *__restrict tilingData);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<KV_T> &kvMergeGm, const GlobalTensor<KV_T> &oriKvGm,
                                                const GlobalTensor<KV_T> &cmpKvGm,
                                                const GlobalTensor<int32_t> &oriBlockTableGm,
                                                const GlobalTensor<int32_t> &cmpBlockTableGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm,
                                                GlobalTensor<int32_t> topKGm, GlobalTensor<T> sinksGm,
                                                GlobalTensor<T> softmaxLseGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void CopySinksIn();
    __aicore__ inline void SliceAndContactSinksValue(uint32_t nIdx, uint32_t dealRowCount);
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    // ================================Vector0==========================================
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo &runInfo);
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo &runInfo);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);

    __aicore__ inline void ElewiseCompute(const RunInfo &info, const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                          uint32_t columnCount);
    __aicore__ inline void ProcessLse(const RunInfo &info, const MSplitInfo &mSplitInfo);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t mStartRow,
                                            uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);

    // BLOCK and REPEAT sizes in bytes.
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // Number of FP32 elements in a BLOCK and REPEAT.
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // The repeat stride must not exceed 256.
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    static constexpr bool PAGE_ATTENTION = SMLAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SMLAT::templateMode;
    static constexpr bool FLASH_DECODE = SMLAT::flashDecode;
    static constexpr SMLA_LAYOUT LAYOUT_T = SMLAT::layout;
    static constexpr SMLA_LAYOUT KV_LAYOUT_T = SMLAT::kvLayout;
    static constexpr bool HEAD_RATIO_ONE = SMLAT::headRatioOne;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 3;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_INPUT_BUF2_PONG_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;
    static constexpr uint64_t SYNC_INPUT_V0BUF_FLAG = 7;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t INPUT2_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_16K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_1K;
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T); // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                               // 32/4=8
    static constexpr uint32_t MAX_N1_SIZE = 128U;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr SINKS_T R0 = 1.0f;

    const SmlaVaTilingData *__restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo constInfo = {};

    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<T> sinksGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<T> softmaxLseGm;

    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<KV_T> oriKvGm_;
    GlobalTensor<KV_T> cmpKvGm_;
    GlobalTensor<int32_t> oriBlockTableGm_;
    GlobalTensor<int32_t> cmpBlockTableGm_;

    // ================================Local buffers====================================
    TBuf<> inputBuff1;  // 32K
    TBuf<> inputBuff2;  // 16K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 32K

    TBuf<> tmpBuff1;        // 32K
    TBuf<> v0ValidSizeBuff; // 8K

    TBuf<> sinksBuff;     // 1K
    TBuf<> sinksBrcbBuff; // 12K

    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;
    LocalTensor<SINKS_T> sinksUb;
    LocalTensor<SINKS_T> sinksBrcbUb;

    uint32_t mergeMte3Idx = 0;
};

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, ConstInfo::BUFFER_SIZE_BYTE_16K * 2); // 2:pingpong
    pipe->InitBuffer(outputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    if (constInfo.returnSoftmaxLse) {
        pipe->InitBuffer(outputBuff2, ConstInfo::BUFFER_SIZE_BYTE_1K);
    }

    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::BUFFER_SIZE_BYTE_8K);

    // M_MAX = 512/2vector = 256, 256 * sizeof(T) * N_Buffer

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_1K);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_1K);

    pipe->InitBuffer(sinksBuff, MAX_N1_SIZE * sizeof(SINKS_T));
    // Allocate 256+N1 elements; 256 is the maximum VEC tile size along M.
    pipe->InitBuffer(sinksBrcbBuff, MAX_N1_SIZE * sizeof(SINKS_T) * BLOCK_ELEMENT_NUM * 3U);

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff2.Get<KV_T>();

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();

    sinksUb = sinksBuff.Get<SINKS_T>();
    sinksBrcbUb = sinksBrcbBuff.Get<SINKS_T>();
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::InitParams(const struct ConstInfo &constInfo,
                                                          const SmlaVaTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::InitVec0GlobalTensor(const GlobalTensor<KV_T> &kvMergeGm,
                                                                    const GlobalTensor<KV_T> &oriKvGm,
                                                                    const GlobalTensor<KV_T> &cmpKvGm,
                                                                    const GlobalTensor<int32_t> &oriBlockTableGm,
                                                                    const GlobalTensor<int32_t> &cmpBlockTableGm)
{
    this->kvMergeGm_ = kvMergeGm;
    this->oriKvGm_ = oriKvGm;
    this->cmpKvGm_ = cmpKvGm;
    this->oriBlockTableGm_ = oriBlockTableGm;
    this->cmpBlockTableGm_ = cmpBlockTableGm;
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm, GlobalTensor<int32_t> actualSeqLengthsQGm,
    GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<int32_t> topKGm, GlobalTensor<SINKS_T> sinksGm,
    GlobalTensor<T> softmaxLseGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->topkGm_ = topKGm;
    this->sinksGm = sinksGm;
    this->softmaxLseGm = softmaxLseGm;
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                    GlobalTensor<UPDATE_T> vec2ResGm,
                                                                    GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                    GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::CopySinksIn()
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = constInfo.qHeadNum * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(sinksUb, sinksGm, dataCopyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    uint32_t repeatTimes = (constInfo.qHeadNum + BLOCK_ELEMENT_NUM - 1U) / BLOCK_ELEMENT_NUM; // Process eight data blocks at a time.
    Brcb(sinksBrcbUb, sinksUb, repeatTimes, {1, BLOCK_ELEMENT_NUM});
    PipeBarrier<PIPE_V>();

    DataCopyParams repeatParams;
    repeatParams.blockCount = 1; // Copy enough blocks to exceed the M-axis tile size of a single vector core; each vector core gets 256 rows.
    repeatParams.blockLen = constInfo.qHeadNum;
    repeatParams.srcStride = 0U;
    repeatParams.dstStride = 0U;
    for (uint32_t i = 1U; i <= 256U / constInfo.qHeadNum; i++) {
        DataCopy(sinksBrcbUb[constInfo.qHeadNum * BLOCK_ELEMENT_NUM * i], sinksBrcbUb, repeatParams);
    }
    PipeBarrier<PIPE_V>();
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::SliceAndContactSinksValue(uint32_t nIdx, uint32_t dealRowCount)
{
    // WholeReduceMax supports repeatTimes in (0, 255), so split the work into calls with repeatTime=128.
    uint32_t repeatTimesOnce = 128;
    uint32_t loopTimes = (dealRowCount + repeatTimesOnce - 1) / repeatTimesOnce;
    uint32_t repeatTimes = repeatTimesOnce;

    for (uint32_t loop = 0; loop < loopTimes; ++loop) {
        if (loop == loopTimes - 1) {
            repeatTimes = dealRowCount - loop * repeatTimesOnce;
        }
        WholeReduceMax(softmaxMaxDefaultUb[loop * repeatTimesOnce],
                       sinksBrcbUb[(nIdx + loop * repeatTimesOnce) * BLOCK_ELEMENT_NUM],
                       BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM, repeatTimes, 1, 0, 1, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
    }
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::InitSoftmaxDefaultBuffer()
{
    CopySinksIn();
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
    Duplicate(softmaxSumDefaultUb, R0, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::ElewiseCompute(const RunInfo &info, const LocalTensor<T> &mmResUb,
                                                              uint32_t dealRowCount, uint32_t columnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.softmaxScale), dealRowCount * columnCount);
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::ProcessLse(const RunInfo &info, const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t lseOffset;
    if (constInfo.outputLayout == SMLA_LAYOUT::TND) {
        uint32_t tBase = actualSeqLengthsQGm.GetValue(info.bIdx);
        lseOffset = (tBase + info.s1Idx) * constInfo.gSize +               // Offset along T and S1.
                    info.n2IdxReal * constInfo.qSeqSize * constInfo.gSize; // Offset along N2.
    } else if (constInfo.outputLayout == SMLA_LAYOUT::BSND) {
        lseOffset = info.bIdx * constInfo.qSeqSize * constInfo.kvHeadNum * constInfo.gSize + // Offset along B.
                    info.n2IdxReal * constInfo.qSeqSize * constInfo.gSize +                  // Offset along N2.
                    info.s1Idx * constInfo.gSize;                                            // Offset along S1.
    }
    lseOffset = lseOffset + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    auto sumTensor = softmaxSumUb[softmaxOffset];
    auto maxTensor = softmaxMaxUb[softmaxOffset];
    auto outLSETensor = outputBuff2.Get<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = mSplitInfo.vecDealM * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    PipeBarrier<PIPE_V>();
    Log(outLSETensor, sumTensor, mSplitInfo.vecDealM);
    PipeBarrier<PIPE_V>();
    Add(outLSETensor, outLSETensor, maxTensor, mSplitInfo.vecDealM);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

    DataCopyPad(softmaxLseGm[lseOffset], outLSETensor, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::SoftmaxFlashV2Compute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                                     LocalTensor<T> &mmResUb,
                                                                     LocalTensor<uint8_t> &softmaxTmpUb,
                                                                     uint32_t startRow, uint32_t dealRowCount,
                                                                     uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb[startRow];
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
    }
    if (actualColumnCount != 0) {
        SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, SMLA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        uint32_t dealRowCountAlign = SMLAAlign(dealRowCount, FP32_BLOCK_ELEMENT_NUM);
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCountAlign);
        PipeBarrier<PIPE_V>();
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCountAlign);
    }
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                                    uint32_t startRow, uint32_t dealRowCount,
                                                                    uint32_t columnCount, uint32_t loopId)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(info, mmResUb, dealRowCount, columnCount);

    PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount,
                          info.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    LocalTensor<KV_T> tmpMMResCastTensor = outputBuff1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    pingpongFlag ^= 1; // Toggle ping-pong buffer 0/1.

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = info.actualSingleProcessSInnerSize == 0 ?
                              16 :
                              BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    // 1. Align down to eight elements because UB operations require at least 32 bytes.
    // 2. info.actualSingleProcessSInnerSizeAlign is at most 512; mSplitSize ensures a minimum of 16.
    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;

    uint32_t sinkHeadIdx =
        (info.n2IdxReal * constInfo.gSize + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) % constInfo.qHeadNum;
    SliceAndContactSinksValue(sinkHeadIdx, mSplitInfo.vecDealM);

    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign, i);
    }
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                            int64_t topkGmBaseOffset, const RunInfo &runInfo)
{
    int64_t cmpS2Offset = s2GmOffset;
    int64_t topkGmIdx = cmpS2Offset / constInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= constInfo.sparseBlockCount || s2GmOffset >= runInfo.v0S2DealSize)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>(cmpS2Offset % constInfo.sparseBlockSize);
}

template <typename SMLAT>
__aicore__ inline int64_t SMLAVectorBlock<SMLAT>::GetKeyGmOffset(int64_t realS2Idx, const RunInfo &runInfo,
                                                                 int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyGmOffset = 0;
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        int64_t blkTableIdx = realS2Idx / constInfo.paCmpBlockSize;
        int64_t blkTableOffset = realS2Idx % constInfo.paCmpBlockSize;
        realKeyGmOffset =
            cmpBlockTableGm_.GetValue(runInfo.bIdx * constInfo.cmpMaxBlockNumPerBatch + blkTableIdx) *
                static_cast<int64_t>(constInfo.cmpKvStride0) +
            blkTableOffset * static_cast<int64_t>(constInfo.kvHeadNum) * static_cast<int64_t>(constInfo.headDim);

    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::BSND) {
        int64_t batchStride = (constInfo.cmpKvStride0 == 0) ? static_cast<int64_t>(constInfo.cmpSeqSize) *
                                                                  static_cast<int64_t>(constInfo.kvHeadNum) *
                                                                  static_cast<int64_t>(constInfo.headDim) :
                                                              static_cast<int64_t>(constInfo.cmpKvStride0);
        realKeyGmOffset =
            static_cast<int64_t>(runInfo.bIdx) * batchStride +
            realS2Idx * static_cast<int64_t>(constInfo.kvHeadNum) * static_cast<int64_t>(constInfo.headDim) +
            static_cast<int64_t>(runInfo.n2Idx) * static_cast<int64_t>(constInfo.headDim);
    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        realKeyGmOffset =
            runInfo.tensorCmpBOffset +
            realS2Idx * static_cast<int64_t>(constInfo.kvHeadNum) * static_cast<int64_t>(constInfo.headDim) +
            static_cast<int64_t>(runInfo.n2Idx) * static_cast<int64_t>(constInfo.headDim);
    }
    return realKeyGmOffset;
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                              int64_t realS2Idx, int64_t keyBNBOffset,
                                                              int64_t s2IdLimit, const RunInfo &runInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams;
    intriParams.blockLen = validS2Count * constInfo.headDim * sizeof(KV_T);
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams{false, 0, 0, 0};
    DataCopyPad(
        kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) * constInfo.headDim],
        cmpKvGm_[keyBNBOffset], intriParams, padParams);
    mte2Size += validS2Count;
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                        int64_t realS2Idx1, int64_t realS2Idx2, const RunInfo &runInfo)
{
    int64_t s2IdLimit = runInfo.cmpS2IdLimit;

    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }

    int64_t keySrcStride = 0;
    keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                    constInfo.sparseBlockSize * constInfo.headDim) *
                   sizeof(KV_T);
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 || realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                 realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit)) {
        // For stride overflow, negative strides, or excessive S2 lengths, fall back to two copy instructions
        // because two blocks must be copied.
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, runInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, runInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.headDim * sizeof(KV_T);
        intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams{false, 0, 0, 0};

        int64_t startGmOffset = keyOffset1 > -1 ? keyOffset1 : keyOffset2;
        if (keyOffset2 > -1 && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T) +
                              (mte2Size - mte3Size) * constInfo.headDim],
                    cmpKvGm_[startGmOffset], intriParams, padParams);
        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                 int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                 const RunInfo &runInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = mte2Size - mte3Size;
    dataCopyParams.blockLen = constInfo.headDim * sizeof(KV_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(kvMergeGm_[runInfo.cmpLoop % MERGE_CACHE_GM_BUF_NUM * 512 * 512 +
                           (s2GmStartOffset + mte3Size) * constInfo.headDim],
                kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T)], dataCopyParams);
}

// b s1 k
template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::ProcessVec0L(const RunInfo &runInfo)
{
    int64_t s2ProcessSize = runInfo.v0S2DealSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2 * constInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = runInfo.topKBaseOffset + runInfo.v0S2Start;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    int64_t s2SplitPoint = SMLAAlign(s2Pair, 2) * constInfo.sparseBlockSize;
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : s2SplitPoint;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ? s2SplitPoint : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    // Process two basic blocks.
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
         s2GmOffsetArray += 2 * constInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, runInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
            mergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset, runInfo);
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo);
        if ((mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > 16) ||
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }
    return;
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::ProcessVec1L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
                                                                (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }

        if constexpr (HEAD_RATIO_ONE) {
            CrossCoreWaitFlag<ConstInfo::SMLA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
        } else {
            CrossCoreWaitFlag(constInfo.syncC1V1);
        }
        // vec1 compute
        ProcessVec1SingleBuf(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SMLA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);

        // move lse for flash decode or FA
        if (constInfo.returnSoftmaxLse && info.s2Idx == info.curSInnerLoopTimes - 1) {
            ProcessLse(info, mSplitInfo);
        }
    }
}

template <typename SMLAT>
__aicore__ inline uint64_t SMLAVectorBlock<SMLAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    return 0;
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::ProcessVec2SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    ProcessVec2Inner(info, mSplitInfo, 0, mSplitInfo.vecDealM);
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::ProcessVec2L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
                                                                (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }
        if constexpr (HEAD_RATIO_ONE) {
            CrossCoreWaitFlag<ConstInfo::SMLA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
        } else {
            CrossCoreWaitFlag(constInfo.syncC2V2);
        }
        ProcessVec2SingleBuf(info, mSplitInfo);
    }
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::ProcessVec2Inner(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                                uint32_t mStartRow, uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * mSplitSize + mStartRow, dealSize, constInfo.headDim,
                             constInfo.headDim);
    }
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                                 uint32_t wsMStart, uint32_t dealRowCount,
                                                                 uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset =
        accumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim +              // taskoffset
        info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim + // Offset for the split.
        wsMStart * actualColumnCount;                                                                 // Offset along M.
    GlobalTensor<T> dst = accumOutGm[offset];
    if (info.actualSingleProcessSInnerSize == 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, dealRowCount * actualColumnCount, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb,
                                                                    uint32_t wsMStart, uint32_t dealRowCount,
                                                                    uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[info.attenOutOffset + wsMStart * actualColumnCount], attenOutUb, dataCopyParams);
    return;
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                                  uint32_t wsMStart, uint32_t dealRowCount,
                                                                  uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // Use round-to-nearest-even for BF16.
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                              uint32_t wsMStart, uint32_t dealRowCount,
                                                              uint32_t columnCount, uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                                    uint32_t startRow, uint32_t dealRowCount,
                                                                    uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t mStart = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + mStart * columnCount;
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);
    DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    pingpongFlag ^= 1; // Toggle ping-pong buffer 0/1.

    uint32_t inOutBaseOffset = mStart * columnCount;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;

    if constexpr (!HEAD_RATIO_ONE) {
        if (!info.isFirstSInnerLoop) {
            event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);

            LocalTensor<MM2_OUT_T> bmm2ResPreUb = inputBuff1.Get<MM2_OUT_T>();
            bmm2ResPreUb = bmm2ResPreUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

            uint64_t vec2ResGmOffset =
                ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
            DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);

            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

            uint32_t idx = info.loop % (constInfo.preLoadNum);
            LocalTensor<T> expUb = v0ValidSizeBuff.Get<T>()[384];
            Brcb(expUb, softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (dealRowCount + 7) / 8,
                 {1, 8});
            PipeBarrier<PIPE_V>();

            RowMuls(bmm2ResPreUb, bmm2ResPreUb, expUb, dealRowCount, columnCount, actualColumnCount);
            AscendC::PipeBarrier<PIPE_V>();
            Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
            AscendC::PipeBarrier<PIPE_V>();

            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
            pingpongFlag ^= 1;
        }

        if (info.isLastS2Loop) {
            uint32_t idx = info.loop % (constInfo.preLoadNum);
            LocalTensor<T> tmpSumUb = v0ValidSizeBuff.Get<T>()[384];
            Brcb(tmpSumUb, softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset],
                 (dealRowCount + 7) / 8, {1, 8});
            PipeBarrier<PIPE_V>();
            RowDivs(bmm2ResUb, bmm2ResUb, tmpSumUb, dealRowCount, columnCount, actualColumnCount);
            PipeBarrier<PIPE_V>();
            Bmm2ResCopyOut(info, bmm2ResUb, mStart, dealRowCount, columnCount, actualColumnCount);
        } else {
            LocalTensor<T> outUb = outputBuff1.Get<T>();
            WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
            DataCopy(outUb, bmm2ResUb, dealRowCount * columnCount);
            SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
            WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
            uint64_t vec2ResGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
            DataCopy(vec2ResGm[vec2ResGmOffset], outUb, vec2ComputeSize);
            SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        }
    } else {
        // Update intermediate results on every iteration except the first.
        if (!info.isFirstSInnerLoop) {
            event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);

            LocalTensor<MM2_OUT_T> bmm2ResPreUb = inputBuff1.Get<MM2_OUT_T>();
            bmm2ResPreUb = bmm2ResPreUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

            uint64_t accumGmOffset =
                ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
            DataCopy(bmm2ResPreUb, mm2ResGm[accumGmOffset], vec2ComputeSize);

            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

            uint32_t idx = info.loop % (constInfo.preLoadNum);
            LocalTensor<T> expUb = v0ValidSizeBuff.Get<T>()[384]; // sumUb uses temporary memory: 16 * 32 bytes = 512 bytes.
            Brcb(expUb, softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (dealRowCount + 7) / 8,
                 {1, 8});
            PipeBarrier<PIPE_V>();

            RowMuls(bmm2ResPreUb, bmm2ResPreUb, expUb, dealRowCount, columnCount, actualColumnCount);
            AscendC::PipeBarrier<PIPE_V>();
            Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
            AscendC::PipeBarrier<PIPE_V>();

            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
            pingpongFlag ^= 1; // Toggle ping-pong buffer 0/1.
        }

        // Output the result on the final iteration; otherwise store intermediate results in workspace.
        if (info.isLastS2Loop) {
            uint32_t idx = info.loop % (constInfo.preLoadNum);
            LocalTensor<T> tmpSumUb = v0ValidSizeBuff.Get<T>()[384]; // sumUb uses temporary memory: 16 * 32 bytes = 512 bytes.
            Brcb(tmpSumUb, softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset],
                 (dealRowCount + 7) / 8, {1, 8});
            PipeBarrier<PIPE_V>();
            RowDivs(bmm2ResUb, bmm2ResUb, tmpSumUb, dealRowCount, columnCount, actualColumnCount);
            PipeBarrier<PIPE_V>();
            Bmm2ResCopyOut(info, bmm2ResUb, mStart, dealRowCount, columnCount, actualColumnCount);
        } else if (!info.isFirstSInnerLoop) {
            LocalTensor<T> outUb = outputBuff1.Get<T>();
            WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
            DataCopy(outUb, bmm2ResUb, dealRowCount * columnCount);
            SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
            WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
            uint64_t accumGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
            DataCopy(mm2ResGm[accumGmOffset], outUb, vec2ComputeSize);
            SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
            WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
            SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        }
    }
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub,
                                                       LocalTensor<float> src1Ub, uint32_t dealRowCount,
                                                       uint32_t columnCount, uint32_t actualColumnCount)
{
    // Divide each row by a shared scalar.
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // Consecutive column repeats start dtypeMask=64 elements, or eight blocks, apart.
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // Consecutive column repeats start dtypeMask=64 elements, or eight blocks, apart.
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

template <typename SMLAT>
__aicore__ inline void SMLAVectorBlock<SMLAT>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub,
                                                       LocalTensor<T> src1Ub, uint32_t dealRowCount,
                                                       uint32_t columnCount, uint32_t actualColumnCount)
{
    // Multiply each row by a shared scalar.
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // Each repeat can read at most 256 contiguous bytes.
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // Each operation reads at most 256 contiguous bytes, processing 256/sizeof(dType) elements.
    // Use dLoop iterations along columns, processing eight columns per iteration.
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256 because the uint8 src0RepStride limits the spacing to at most 256 data blocks.
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // Repeat along columns when that requires fewer iterations than repeating along rows; otherwise repeat along rows.
        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // Consecutive column repeats start dtypeMask=64 elements, or eight blocks, apart.
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8; // Consecutive column repeats start dtypeMask=64 elements, or eight blocks, apart.
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // For the final [dealRowCount, dRemain] * [dealRowCount, blockElementNum] operation, compute only valid elements.
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // Each repeat processes 256 bytes, exactly eight data blocks.
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // Process one row at a time, for dealRowCount rows.
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // Process dLoop repeats per row; each repeat handles 256/block_size data blocks.
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            // Process the tail block of a row.
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}
} // namespace SMLAKernel
#endif // SMLA_VA_CSA_BLOCK_VECTOR_H
