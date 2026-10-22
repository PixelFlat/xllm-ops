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
 * \file smla_va_swa_block_cube.h
 * \brief use 7 buffer for matmul l1, better pipeline
 */
#ifndef SMLA_VA_SWA_BLOCK_CUBE_H
#define SMLA_VA_SWA_BLOCK_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "smla_va_common_arch22.h"

namespace SMLAKernel {
template <typename SMLAT>
class SWACubeBlock {
public:
    // Use float for intermediate calculations in high-precision mode.
    using T = float;
    using Q_T = typename SMLAT::queryType;
    using KV_T = typename SMLAT::kvType;
    using OUT_T = typename SMLAT::outputType;
    using MM_OUT_T = T;

    __aicore__ inline SWACubeBlock(){};
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<KV_T> oriKvGm,
                                               GlobalTensor<KV_T> cmpKV, GlobalTensor<MM_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<MM_OUT_T> mm2ResGm,
                                               GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void InitPageAttentionInfo(GlobalTensor<KV_T> oriKvGm, GlobalTensor<KV_T> kvMergeGm,
                                                 GlobalTensor<int32_t> oriBlockTableGm,
                                                 GlobalTensor<int32_t> cmpBlockTableGm);
    __aicore__ inline void InitBuffers(TPipe *pipe);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const RunInfo &info, const MSplitInfo mSplitInfo);
    __aicore__ inline void ComputeMm2(const RunInfo &info, const MSplitInfo mSplitInfo);

private:
    static constexpr bool PAGE_ATTENTION = SMLAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SMLAT::templateMode;
    static constexpr bool FLASH_DECODE = SMLAT::flashDecode;
    static constexpr SMLA_LAYOUT LAYOUT_T = SMLAT::layout;
    static constexpr SMLA_LAYOUT KV_LAYOUT_T = SMLAT::kvLayout;

    static constexpr uint32_t M_SPLIT_SIZE = 128;     // Tile size along M.
    static constexpr uint32_t N_SPLIT_SIZE = 128;     // Tile size along N.
    static constexpr uint32_t K_L0_SPLIT_SIZE = 128;  // L0 tile size along K.
    static constexpr uint32_t K_L1_SPLIT_SIZE = 256;  // L1 tile size along K.
    static constexpr uint32_t N_WORKSPACE_SIZE = 512; // Tile size along N.
    static constexpr uint32_t MERGE_CACHE_GM_BUF_NUM = 3;
    static constexpr uint32_t D_SPLIT_SIZE = 256; // Tile size along D.

    static constexpr uint32_t L1_BLOCK_SIZE = (64 * 512 * sizeof(Q_T));
    static constexpr uint32_t L1_BLOCK_OFFSET = 64 * 512;

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    // mte2 <> mte1 EventID
    // Three L1 buffers use three event IDs.
    static constexpr uint32_t L1_EVENT0 = EVENT_ID2;
    static constexpr uint32_t L1_EVENT1 = EVENT_ID3;
    static constexpr uint32_t L1_EVENT2 = EVENT_ID4;
    static constexpr uint32_t L1_EVENT3 = EVENT_ID5;
    static constexpr uint32_t L1_EVENT4 = EVENT_ID6;
    static constexpr uint32_t L1_EVENT5 = EVENT_ID7;
    static constexpr uint32_t L1_EVENT6 = EVENT_ID1;

    // m <> mte1 EventID
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
    static constexpr uint32_t mte21QPIds[4] = {L1_EVENT0, L1_EVENT1, L1_EVENT2, L1_EVENT3}; // Reuse MTE12 events.
    static constexpr uint32_t mte21KVIds[3] = {L1_EVENT4, L1_EVENT5, L1_EVENT6};

    ConstInfo constInfo{};

    // Track the three L1 buffers.
    uint32_t qpL1BufIter = 0;
    uint32_t kvL1BufIter = -1;
    uint32_t abL0BufIter = 0;
    uint32_t cL0BufIter = 0;

    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> oriKvGm;
    GlobalTensor<KV_T> cmpKvGm;

    // mm2
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<OUT_T> attentionOutGm;

    // block_table
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;

    TBuf<TPosition::A1> bufQPL1;
    TBuf<TPosition::A1> bufKVL1;
    TBuf<TPosition::A2> tmpBufL0A;
    TBuf<TPosition::B2> tmpBufL0B;
    TBuf<TPosition::CO1> tmpBufL0C;

    LocalTensor<Q_T> l1QPTensor;
    LocalTensor<Q_T> l1KVTensor;
    LocalTensor<KV_T> aL0TensorPingPong;
    LocalTensor<KV_T> bL0TensorPingPong;
    LocalTensor<MM_OUT_T> cL0TensorPingPong;

    // L0AB m <> mte1 EventID
    __aicore__ inline uint32_t Mte1MmABEventId(uint32_t idx)
    {
        return (L0AB_EVENT0 + idx);
    }

    __aicore__ inline uint32_t GetQPL1RealIdx(uint32_t mIdx, uint32_t k1Idx)
    {
        uint32_t idxMap[] = {0, 2}; // Keep blocks 0/1 and 2/3 adjacent so addresses within the same M block are contiguous.
        return idxMap[mIdx % 2] + k1Idx;
    }

    __aicore__ inline void CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info, uint32_t mSeqIdx,
                                          uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info, uint32_t mSeqIdx,
                                          uint32_t subMSizeAct, uint32_t nSize, uint32_t nOffset);
    __aicore__ inline void LoadDataMm1A(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor, uint32_t idx,
                                        uint32_t kSplitSize, uint32_t mSize, uint32_t kSize);
    __aicore__ inline void LoadDataMm1B(LocalTensor<KV_T> &bL0Tensor, LocalTensor<KV_T> &bL1Tensor, uint32_t idx,
                                        uint32_t kSplitSize, uint32_t kSize, uint32_t nSize);
};

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<KV_T> oriKvGm,
                                                                GlobalTensor<KV_T> cmpKvGm,
                                                                GlobalTensor<MM_OUT_T> mm1ResGm)
{
    // mm1
    this->queryGm = queryGm;
    this->oriKvGm = oriKvGm;
    if constexpr (TEMPLATE_MODE == HCA_TEMPLATE) {
        this->cmpKvGm = cmpKvGm;
    }
    this->mm1ResGm = mm1ResGm;
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm,
                                                                GlobalTensor<MM_OUT_T> mm2ResGm,
                                                                GlobalTensor<OUT_T> attentionOutGm)
{
    // mm2
    this->vec1ResGm = vec1ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::InitPageAttentionInfo(GlobalTensor<KV_T> oriKvGm,
                                                                  GlobalTensor<KV_T> kvMergeGm,
                                                                  GlobalTensor<int32_t> oriBlockTableGm,
                                                                  GlobalTensor<int32_t> cmpBlockTableGm)
{
    this->oriKvGm = oriKvGm;
    this->kvMergeGm_ = kvMergeGm;
    this->oriBlockTableGm = oriBlockTableGm;
    if constexpr (TEMPLATE_MODE == HCA_TEMPLATE) {
        this->cmpBlockTableGm = cmpBlockTableGm;
    }
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4);
    l1QPTensor = bufQPL1.Get<Q_T>();
    pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3);
    l1KVTensor = bufKVL1.Get<KV_T>();
    // L0A
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2); // 64K
    aL0TensorPingPong = tmpBufL0A.Get<KV_T>();
    // L0B
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2); // 64K
    bL0TensorPingPong = tmpBufL0B.Get<KV_T>();
    // L0C
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2); // 128K
    cL0TensorPingPong = tmpBufL0C.Get<MM_OUT_T>();
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor,
                                                       uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // Number of rows.
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16; // Align to 16 blocks.
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::CopyInMm1AToL1(LocalTensor<KV_T> &l1Tensor, const RunInfo &info,
                                                           uint32_t mSeqIdx, uint32_t mSizeAct, uint32_t headSize,
                                                           uint32_t headOffset)
{
    auto srcGm = queryGm[info.tensorAOffset + mSeqIdx * constInfo.headDim + headOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, headSize, constInfo.headDim);
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::LoadDataMm1A(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor,
                                                         uint32_t idx, uint32_t kSplitSize, uint32_t mSize,
                                                         uint32_t kSize)
{
    LocalTensor<KV_T> srcTensor = aL1Tensor[mSize * kSplitSize * idx];
    LoadData3DParamsV2<KV_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = mSize / 16; // Hin=M1=8
    loadData3DParams.l1W = 16;         // Win=M0
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // Tail data does not affect the sliding-window result.

    // SetLoadToA0Params
    loadData3DParams.mExtension = mSize; // M
    loadData3DParams.kExtension = kSize; // K
    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;
    loadData3DParams.channelSize = kSize; // Cin=K
    LoadData<KV_T, LOAD3DV2_CONFIG>(aL0Tensor, srcTensor, loadData3DParams);
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::LoadDataMm1B(LocalTensor<KV_T> &l0Tensor, LocalTensor<KV_T> &l1Tensor,
                                                         uint32_t idx, uint32_t kSplitSize, uint32_t kSize,
                                                         uint32_t nSize)
{
    // Load the full N dimension.
    LocalTensor<KV_T> srcTensor = l1Tensor[nSize * kSplitSize * idx];
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = (nSize + 15) / 16 * kSize / (32 / sizeof(KV_T));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0Tensor, srcTensor, loadData2DParams);
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info,
                                                           uint32_t mSeqIdx, uint32_t subMSizeAct, uint32_t nSize,
                                                           uint32_t nOffset)
{
    auto srcGm = vec1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                           mSeqIdx * info.actualSingleProcessSInnerSizeAlign + nOffset];
    CopyGmToL1(aL1Tensor, srcGm, subMSizeAct, nSize, info.actualSingleProcessSInnerSizeAlign);
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::ComputeMm1(const RunInfo &info, const MSplitInfo mSplitInfo)
{
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mL1Size = M_SPLIT_SIZE;
    uint32_t mL1SizeAlign = SMLAAlign(M_SPLIT_SIZE, 16);
    uint32_t mL1Loops = CeilDiv(mSize, M_SPLIT_SIZE);

    uint32_t nSize = info.actualSingleProcessSInnerSize;
    uint32_t nL1Size = N_SPLIT_SIZE;
    uint32_t nL1SizeAlign = SMLAAlign(N_SPLIT_SIZE, 16);
    uint32_t nL1Loops = CeilDiv(nSize, N_SPLIT_SIZE);

    uint32_t kSize = 512;
    uint32_t kL1Size = 256;
    uint32_t kL1Loops = 2;
    uint32_t kL0Size = 128;
    uint32_t kL0Loops = CeilDiv(kL1Size, kL0Size);

    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> kTensor;
    uint32_t ka = 0, kb = 0;
    uint32_t copyRowCnt = 0;
    uint32_t copyRowCntTmp = 0;
    // Tile N and K in L1.
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // Tile N in L1: 512/128=4.
        if (nL1 == (nL1Loops - 1)) {
            // Recalculate the tail block size.
            nL1Size = nSize - (nL1Loops - 1) * N_SPLIT_SIZE;
            nL1SizeAlign = SMLAAlign(nL1Size, 16);
        }

        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
            kvL1BufIter++;
            uint32_t kb = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
            // Select the current block along K.
            bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
            uint32_t copyFinishRowCnt = 0;

            if (info.isOriOnly) {
                if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                    uint64_t curS2Offset = static_cast<uint64_t>(info.s2Idx) * constInfo.s2BaseSize +
                                           info.s2StartPoint + nL1 * N_SPLIT_SIZE;
                    uint32_t copyFinishRowCnt = 0;
                    LocalTensor<KV_T> kTensor;
                    uint32_t copyRowCnt = 0;

                    if (constInfo.hasOriSparseIndices) {
                        uint64_t mergeBase = static_cast<uint64_t>(info.cmpLoop % MERGE_CACHE_GM_BUF_NUM) *
                                             N_WORKSPACE_SIZE * constInfo.headDim;
                        uint32_t blockElementCnt = 32U / sizeof(KV_T);
                        for (uint32_t row = 0; row < nL1Size; ++row) {
                            uint64_t mergeOffset = mergeBase +
                                                   static_cast<uint64_t>(nL1 * N_SPLIT_SIZE + row) * constInfo.headDim +
                                                   kL1 * D_SPLIT_SIZE;
                            GlobalTensor<KV_T> mergeSrcGm = kvMergeGm_[mergeOffset];
                            kTensor = bL1Tensor[row * blockElementCnt];
                            DataCopyGmNDToL1<KV_T>(kTensor, mergeSrcGm, 1, nL1SizeAlign, D_SPLIT_SIZE,
                                                   constInfo.headDim);
                        }
                    } else
                        while (copyFinishRowCnt < nL1Size) {
                            // Because of ori_left, even the first transfer may start at a nonzero offset within a PA block.
                            copyRowCnt = constInfo.paOriBlockSize - curS2Offset % constInfo.paOriBlockSize;
                            if (copyFinishRowCnt + copyRowCnt > nL1Size) {
                                copyRowCnt = nL1Size - copyFinishRowCnt;
                            }
                            Position startPos;
                            startPos.bIdx = info.bIdx;
                            startPos.n2Idx = info.n2Idx;
                            startPos.s2Idx = curS2Offset;
                            // TODO: Rename constants 256 and 32 for the seven-buffer scheme.
                            startPos.dIdx = kL1 * 256; // MM1 right matrix: BN2S2D, with D as the untiled K axis; MM2 right matrix: S2 is the K axis and D is tiled.
                            PAShape shape;
                            shape.blockSize = constInfo.paOriBlockSize;
                            shape.headNum = constInfo.kvHeadNum;
                            shape.headDim = constInfo.headDim;
                            shape.kvStride = constInfo.oriKvStride0;
                            shape.actHeadDim = 256;
                            shape.maxblockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
                            shape.copyRowNum = copyRowCnt;
                            shape.copyRowNumAlign = nL1SizeAlign;
                            kTensor = bL1Tensor[copyFinishRowCnt * 16];
                            DataCopyPA<KV_T>(kTensor, oriKvGm, oriBlockTableGm, shape, startPos);
                            // Update loop variables.
                            copyFinishRowCnt += copyRowCnt;
                            curS2Offset += copyRowCnt;
                        }
                } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::BSND) {
                    Nd2NzParams nd2nzPara;
                    nd2nzPara.ndNum = 1;
                    nd2nzPara.nValue = nL1Size;      // Number of rows.
                    nd2nzPara.dValue = D_SPLIT_SIZE; // 256
                    nd2nzPara.srcDValue = constInfo.headDim;
                    nd2nzPara.dstNzC0Stride = nL1SizeAlign;
                    nd2nzPara.dstNzNStride = 1;
                    nd2nzPara.srcNdMatrixStride = 0;
                    nd2nzPara.dstNzMatrixStride = 0;

                    uint32_t headStride = constInfo.headDim;
                    uint32_t seqStride = constInfo.kvHeadNum * constInfo.headDim;
                    uint64_t batchStride = (constInfo.oriKvStride0 == 0) ?
                                               static_cast<uint64_t>(constInfo.kvSeqSize) * seqStride :
                                               constInfo.oriKvStride0;

                    uint64_t curS2 = info.s2StartPoint + nL1 * N_SPLIT_SIZE;
                    uint64_t offset = (uint64_t)info.bIdx * batchStride + (uint64_t)curS2 * seqStride +
                                      (uint64_t)info.n2Idx * headStride + kL1 * D_SPLIT_SIZE;
                    DataCopy(bL1Tensor, oriKvGm[offset], nd2nzPara);
                } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
                    uint64_t curS2Offset =
                        static_cast<uint64_t>(info.s2Idx) * static_cast<uint64_t>(constInfo.s2BaseSize) +
                        info.s2StartPoint + nL1 * N_SPLIT_SIZE;
                    Nd2NzParams nd2nzPara;
                    nd2nzPara.ndNum = 1;
                    nd2nzPara.nValue = nL1Size;
                    nd2nzPara.dValue = constInfo.headDim >> 1;
                    nd2nzPara.srcDValue = constInfo.headDim;
                    nd2nzPara.dstNzC0Stride = nL1SizeAlign;
                    nd2nzPara.dstNzNStride = 1;
                    nd2nzPara.srcNdMatrixStride = 0;
                    nd2nzPara.dstNzMatrixStride = 0;
                    DataCopy(bL1Tensor,
                             oriKvGm[info.tensorBOffset + curS2Offset * constInfo.headDim + kL1 * D_SPLIT_SIZE],
                             nd2nzPara);
                }
            } else if (info.isOriCmpMix) {
                uint32_t cumNl1LoopSize = nL1 * N_SPLIT_SIZE;
                uint32_t oriSizeCur = (info.actualSingleProcessSInnerOriSize < cumNl1LoopSize) ?
                                          0 :
                                          (info.actualSingleProcessSInnerOriSize - cumNl1LoopSize);
                oriSizeCur = (oriSizeCur > nL1Size) ? nL1Size : oriSizeCur;
                uint32_t cmpSizeCur = nL1Size - oriSizeCur;
                cmpSizeCur = (cmpSizeCur < info.actualSingleProcessSInnerCmpSize) ?
                                 cmpSizeCur :
                                 info.actualSingleProcessSInnerCmpSize;
                if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                    LocalTensor<KV_T> kTensor;
                    uint32_t copyRowCnt = 0;

                    if (oriSizeCur > 0) {
                        uint32_t copyFinishRowCnt = 0;
                        uint64_t curS2Offset = info.s2StartPoint + nL1 * N_SPLIT_SIZE;
                        while (copyFinishRowCnt < oriSizeCur) {
                            // Because of ori_left, even the first transfer may start at a nonzero offset within a PA block.
                            copyRowCnt = constInfo.paOriBlockSize - curS2Offset % constInfo.paOriBlockSize;
                            if (copyFinishRowCnt + copyRowCnt > oriSizeCur) {
                                copyRowCnt = oriSizeCur - copyFinishRowCnt;
                            }
                            Position startPos;
                            startPos.bIdx = info.bIdx;
                            startPos.n2Idx = info.n2Idx;
                            startPos.s2Idx = curS2Offset;
                            // TODO: Rename constants 256 and 32 for the seven-buffer scheme.
                            startPos.dIdx = kL1 * 256; // MM1 right matrix: BN2S2D, with D as the untiled K axis; MM2 right matrix: S2 is the K axis and D is tiled.
                            PAShape shape;
                            shape.blockSize = constInfo.paOriBlockSize;
                            shape.headNum = constInfo.kvHeadNum;
                            shape.headDim = constInfo.headDim;
                            shape.kvStride = constInfo.oriKvStride0;
                            shape.actHeadDim = 256;
                            shape.maxblockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
                            shape.copyRowNum = copyRowCnt;
                            shape.copyRowNumAlign = nL1SizeAlign;
                            kTensor = bL1Tensor[copyFinishRowCnt * 16];
                            DataCopyPA<KV_T>(kTensor, oriKvGm, oriBlockTableGm, shape, startPos);
                            // Update loop variables.
                            copyFinishRowCnt += copyRowCnt;
                            curS2Offset += copyRowCnt;
                        }
                    }
                    if (cmpSizeCur > 0) {
                        uint32_t cmpMixsizeCur = N_SPLIT_SIZE - info.actualSingleProcessSInnerOriSize % N_SPLIT_SIZE;
                        uint32_t oriOnlyLoopTimes = info.actualSingleProcessSInnerOriSize / N_SPLIT_SIZE;
                        uint32_t cmpLoopTimes = nL1 - oriOnlyLoopTimes;
                        uint64_t curS2Offset =
                            (cmpLoopTimes == 0) ?
                                0 :
                                (cmpMixsizeCur + static_cast<uint64_t>(cmpLoopTimes - 1) * N_SPLIT_SIZE);
                        uint32_t copyFinishRowCnt = 0;
                        while (copyFinishRowCnt < cmpSizeCur) {
                            // Because of ori_left, even the first transfer may start at a nonzero offset within a PA block.
                            copyRowCnt = constInfo.paCmpBlockSize - curS2Offset % constInfo.paCmpBlockSize;
                            if (copyFinishRowCnt + copyRowCnt > cmpSizeCur) {
                                copyRowCnt = cmpSizeCur - copyFinishRowCnt;
                            }
                            Position startPos;
                            startPos.bIdx = info.bIdx;
                            startPos.n2Idx = info.n2Idx;
                            startPos.s2Idx = curS2Offset;
                            // TODO: Rename constants 256 and 32 for the seven-buffer scheme.
                            startPos.dIdx = kL1 * 256; // MM1 right matrix: BN2S2D, with D as the untiled K axis; MM2 right matrix: S2 is the K axis and D is tiled.
                            PAShape shape;
                            shape.blockSize = constInfo.paCmpBlockSize;
                            shape.headNum = constInfo.kvHeadNum;
                            shape.headDim = constInfo.headDim;
                            shape.kvStride = constInfo.cmpKvStride0;
                            shape.actHeadDim = 256;
                            shape.maxblockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
                            shape.copyRowNum = copyRowCnt;
                            shape.copyRowNumAlign = nL1SizeAlign;
                            kTensor = bL1Tensor[copyFinishRowCnt * 16 + oriSizeCur * 16];
                            DataCopyPA<KV_T>(kTensor, cmpKvGm, cmpBlockTableGm, shape, startPos);
                            // Update loop variables.
                            copyFinishRowCnt += copyRowCnt;
                            curS2Offset += copyRowCnt;
                        }
                    }
                } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::BSND) {
                    Nd2NzParams nd2nzPara;
                    nd2nzPara.ndNum = 1;
                    nd2nzPara.nValue = nL1Size;      // Number of rows.
                    nd2nzPara.dValue = D_SPLIT_SIZE; // 256
                    nd2nzPara.srcDValue = constInfo.headDim;
                    nd2nzPara.dstNzC0Stride = nL1SizeAlign;
                    nd2nzPara.dstNzNStride = 1;
                    nd2nzPara.srcNdMatrixStride = 0;
                    nd2nzPara.dstNzMatrixStride = 0;

                    uint32_t headStride = constInfo.headDim;
                    uint32_t seqStride = constInfo.kvHeadNum * constInfo.headDim;
                    if (oriSizeCur > 0) {
                        nd2nzPara.nValue = oriSizeCur;
                        uint64_t batchStride = (constInfo.oriKvStride0 == 0) ?
                                                   static_cast<uint64_t>(constInfo.kvSeqSize) * seqStride :
                                                   constInfo.oriKvStride0;
                        uint64_t curS2 = info.s2StartPoint + nL1 * N_SPLIT_SIZE;
                        uint64_t offset = (uint64_t)info.bIdx * batchStride + (uint64_t)curS2 * seqStride +
                                          (uint64_t)info.n2Idx * headStride + kL1 * D_SPLIT_SIZE;
                        DataCopy(bL1Tensor, oriKvGm[offset], nd2nzPara);
                    }
                    if (cmpSizeCur > 0) {
                        nd2nzPara.nValue = cmpSizeCur;
                        uint64_t batchStride = (constInfo.cmpKvStride0 == 0) ?
                                                   static_cast<uint64_t>(constInfo.cmpSeqSize) * seqStride :
                                                   constInfo.cmpKvStride0;
                        uint32_t cmpMixsizeCur = N_SPLIT_SIZE - info.actualSingleProcessSInnerOriSize % N_SPLIT_SIZE;
                        uint32_t oriOnlyLoopTimes = info.actualSingleProcessSInnerOriSize / N_SPLIT_SIZE;
                        uint32_t cmpLoopTimes = nL1 - oriOnlyLoopTimes;
                        uint64_t curS2 = (cmpLoopTimes == 0) ? 0 : (cmpMixsizeCur + (cmpLoopTimes - 1) * N_SPLIT_SIZE);
                        uint64_t offset = (uint64_t)info.bIdx * batchStride + curS2 * seqStride +
                                          (uint64_t)info.n2Idx * headStride + kL1 * D_SPLIT_SIZE;
                        DataCopy(bL1Tensor[oriSizeCur * 16], cmpKvGm[offset], nd2nzPara);
                    }
                } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
                    Nd2NzParams nd2nzPara;
                    nd2nzPara.ndNum = 1;
                    nd2nzPara.nValue = nL1Size;
                    nd2nzPara.dValue = constInfo.headDim >> 1;
                    nd2nzPara.srcDValue = constInfo.headDim;
                    nd2nzPara.dstNzC0Stride = nL1SizeAlign;
                    nd2nzPara.dstNzNStride = 1;
                    nd2nzPara.srcNdMatrixStride = 0;
                    nd2nzPara.dstNzMatrixStride = 0;

                    if (oriSizeCur > 0) {
                        nd2nzPara.nValue = oriSizeCur;
                        uint64_t curS2Offset = info.s2StartPoint + nL1 * N_SPLIT_SIZE;
                        DataCopy(bL1Tensor,
                                 oriKvGm[info.tensorBOffset + curS2Offset * constInfo.headDim + kL1 * D_SPLIT_SIZE],
                                 nd2nzPara);
                    }
                    if (cmpSizeCur > 0) {
                        nd2nzPara.nValue = cmpSizeCur;
                        uint32_t cmpMixsizeCur = N_SPLIT_SIZE - info.actualSingleProcessSInnerOriSize % N_SPLIT_SIZE;
                        uint32_t oriOnlyLoopTimes = info.actualSingleProcessSInnerOriSize / N_SPLIT_SIZE;
                        uint32_t cmpLoopTimes = nL1 - oriOnlyLoopTimes;
                        uint64_t curS2Offset =
                            (cmpLoopTimes == 0) ?
                                0 :
                                (cmpMixsizeCur + static_cast<uint64_t>(cmpLoopTimes - 1) * N_SPLIT_SIZE);
                        DataCopy(bL1Tensor[oriSizeCur * 16],
                                 cmpKvGm[info.tensorCmpBOffset + curS2Offset * constInfo.headDim + kL1 * D_SPLIT_SIZE],
                                 nd2nzPara);
                    }
                }
            } else {
                if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                    uint64_t curS2Offset = static_cast<uint64_t>(info.relativeS2Idx) * constInfo.s2BaseSize +
                                           info.s2StartPoint + nL1 * N_SPLIT_SIZE;
                    while (copyFinishRowCnt < nL1Size) {
                        // Because of ori_left, even the first transfer may start at a nonzero offset within a PA block.
                        copyRowCnt = constInfo.paCmpBlockSize - curS2Offset % constInfo.paCmpBlockSize;
                        if (copyFinishRowCnt + copyRowCnt > nL1Size) {
                            copyRowCnt = nL1Size - copyFinishRowCnt;
                        }

                        Position startPos;
                        startPos.bIdx = info.bIdx;
                        startPos.n2Idx = info.n2Idx;
                        startPos.s2Idx = curS2Offset;
                        // TODO: Rename constants 256 and 32 for the seven-buffer scheme.
                        startPos.dIdx = kL1 * 256; // MM1 right matrix: BN2S2D, with D as the untiled K axis; MM2 right matrix: S2 is the K axis and D is tiled.

                        PAShape shape;
                        shape.blockSize = constInfo.paCmpBlockSize;
                        shape.headNum = constInfo.kvHeadNum;
                        shape.headDim = constInfo.headDim;
                        shape.kvStride = constInfo.cmpKvStride0;
                        shape.actHeadDim = 256;
                        shape.maxblockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
                        shape.copyRowNum = copyRowCnt;
                        shape.copyRowNumAlign = nL1SizeAlign;
                        kTensor = bL1Tensor[copyFinishRowCnt * 16];
                        DataCopyPA<KV_T>(kTensor, cmpKvGm, cmpBlockTableGm, shape, startPos);
                        // Update loop variables.
                        copyFinishRowCnt += copyRowCnt;
                        curS2Offset += copyRowCnt;
                    }
                } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::BSND) {
                    Nd2NzParams nd2nzPara;
                    nd2nzPara.ndNum = 1;
                    nd2nzPara.nValue = nL1Size;      // Number of rows.
                    nd2nzPara.dValue = D_SPLIT_SIZE; // 256
                    nd2nzPara.srcDValue = constInfo.headDim;
                    nd2nzPara.dstNzC0Stride = nL1SizeAlign;
                    nd2nzPara.dstNzNStride = 1;
                    nd2nzPara.srcNdMatrixStride = 0;
                    nd2nzPara.dstNzMatrixStride = 0;

                    uint32_t headStride = constInfo.headDim;
                    uint32_t seqStride = constInfo.kvHeadNum * constInfo.headDim;
                    uint64_t batchStride = (constInfo.cmpKvStride0 == 0) ?
                                               static_cast<uint64_t>(constInfo.cmpSeqSize) * seqStride :
                                               constInfo.cmpKvStride0;

                    uint64_t curS2 = static_cast<uint64_t>(info.relativeS2Idx) * constInfo.s2BaseSize +
                                     info.s2StartPoint + nL1 * N_SPLIT_SIZE;
                    uint64_t offset = (uint64_t)info.bIdx * batchStride + (uint64_t)curS2 * seqStride +
                                      (uint64_t)info.n2Idx * headStride + kL1 * D_SPLIT_SIZE;
                    DataCopy(bL1Tensor, cmpKvGm[offset], nd2nzPara);
                } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
                    uint64_t curS2Offset = static_cast<uint64_t>(info.relativeS2Idx) * constInfo.s2BaseSize +
                                           info.s2StartPoint + nL1 * N_SPLIT_SIZE;
                    Nd2NzParams nd2nzPara;
                    nd2nzPara.ndNum = 1;
                    nd2nzPara.nValue = nL1Size;
                    nd2nzPara.dValue = constInfo.headDim >> 1;
                    nd2nzPara.srcDValue = constInfo.headDim;
                    nd2nzPara.dstNzC0Stride = nL1SizeAlign;
                    nd2nzPara.dstNzNStride = 1;
                    nd2nzPara.srcNdMatrixStride = 0;
                    nd2nzPara.dstNzMatrixStride = 0;
                    DataCopy(bL1Tensor,
                             cmpKvGm[info.tensorCmpBOffset + curS2Offset * constInfo.headDim + +kL1 * D_SPLIT_SIZE],
                             nd2nzPara);
                }
            }

            SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            mL1Size = M_SPLIT_SIZE;
            mL1SizeAlign = SMLAAlign(M_SPLIT_SIZE, 16U);
            for (uint32_t mL1 = 0; mL1 < mL1Loops; mL1++) {
                uint32_t aL1PaddingSize = 0; // Align the left matrix to the end so the two 32-KiB regions remain contiguous.
                if (mL1 == (mL1Loops - 1)) {
                    mL1Size = mSize - (mL1Loops - 1) * M_SPLIT_SIZE;
                    mL1SizeAlign = SMLAAlign(mL1Size, 16U);
                    aL1PaddingSize = (M_SPLIT_SIZE - mL1SizeAlign) * 256;
                }
                uint32_t mIdx = qpL1BufIter + mL1;
                ka = GetQPL1RealIdx(mIdx, kL1);
                LocalTensor<Q_T> aL1Tensor = l1QPTensor[ka * L1_BLOCK_OFFSET + (1 - kL1) * aL1PaddingSize];
                if (nL1 == 0) {
                    if (kL1 == 0) {
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]);
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka + 1]);
                        CopyInMm1AToL1(aL1Tensor, info, mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE, mL1Size, 256, 0);
                    } else {
                        LocalTensor<Q_T> qTmpTensor = aL1Tensor;
                        CopyInMm1AToL1(qTmpTensor, info, mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE, mL1Size, 256,
                                       256);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                }
                // Synchronize using unit flags.
                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(cL0BufIter % 2) *
                                      (L0C_PP_SIZE / sizeof(MM_OUT_T))]; // Keep cL0BufIter in step with M.
                for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1A(aL0Tensor, aL1Tensor, kL0, kL0Size, mL1SizeAlign, kL0Size);
                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    if (mL1 == 0) {
                        LoadDataMm1B(bL0Tensor, bL1Tensor, kL0, kL0Size, kL0Size, nL1SizeAlign);
                    }
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));

                    MmadParams mmadParams;
                    mmadParams.m = mL1SizeAlign;
                    mmadParams.n = nL1SizeAlign;
                    mmadParams.k = kL0Size;
                    mmadParams.cmatrixInitVal = (kL1 == 0 && kL0 == 0);
                    mmadParams.cmatrixSource = false;
                    mmadParams.unitFlag =
                        (kL1 == 1 && kL0 == (kL0Loops - 1)) ? 0b11 : 0b10; // Flip the flag on the final accumulation to indicate the result can be transferred out.
                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    abL0BufIter++;
                }

                if (nL1 == (nL1Loops - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]); // Reverse synchronization: MTE1 has finished consuming A in L1.
                }

                if (kL1 == 1) { // Final kL1 iteration.
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    fixParams.srcStride = mL1SizeAlign;
                    // Use nSizeAlign.
                    fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // Stride between adjacent mm1ResGm rows.
                    fixParams.unitFlag = 0b11;
                    fixParams.ndNum = 1; // Output in ND layout.

                    Fixpipe(mm1ResGm[(info.loop % (constInfo.preLoadNum)) * constInfo.mmResUbSize + nL1 * N_SPLIT_SIZE +
                                     (mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE) *
                                         info.actualSingleProcessSInnerSizeAlign],
                            cL0Tensor, fixParams);
                }
                if (mL1Loops == 2) {
                    cL0BufIter++;
                }
            }

            SetFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]); // Reverse synchronization: MTE1 has finished consuming L1.
        }
        if (mL1Loops == 1) {
            cL0BufIter++;
        }
    }
    qpL1BufIter += mL1Loops;
}

template <typename SMLAT>
__aicore__ inline void SWACubeBlock<SMLAT>::ComputeMm2(const RunInfo &info, const MSplitInfo mSplitInfo)
{
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mSizeAlign = (mSize + 16 - 1) / 16;
    uint32_t mL1Loops = (mSize + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;
    uint32_t mL1SizeAlign = M_SPLIT_SIZE; // Aligned to 16.
    uint32_t mL1Size = M_SPLIT_SIZE;      // Actual M size.

    uint32_t nSize = BlockAlign<KV_T>(constInfo.headDim);
    uint32_t nL1Loops = (nSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;
    uint32_t nL1SizeAlign = N_SPLIT_SIZE; // Aligned to 16.
    uint32_t nL1Size = N_SPLIT_SIZE;      // Actual N size.

    uint32_t kSize = info.actualSingleProcessSInnerSize;
    uint32_t kL1Size = K_L1_SPLIT_SIZE;
    uint32_t kL1SizeAlign = SMLAAlign(kL1Size, 16U);
    uint32_t kL1Loops = (kSize + kL1Size - 1) / kL1Size;
    uint32_t kL0Size = K_L0_SPLIT_SIZE;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
    uint32_t kL0SizeAlign = kL0Size;
    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> subvTensor;
    // ka selects one of four left-matrix buffers; kb selects one of three right-matrix buffers.
    uint32_t ka = 0, kb = 0;
    uint32_t mBaseIdx = qpL1BufIter;
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // Tile N in L1.
        if (nL1 == (nL1Loops - 1)) {
            // Tail block.
            nL1Size = nSize - (nL1Loops - 1) * N_SPLIT_SIZE;
            nL1SizeAlign = SMLAAlign(nL1Size, 16U);
        }
        // Use one K-L1 loop, consistent with MM1.
        kL1Size = K_L1_SPLIT_SIZE;
        kL1SizeAlign = SMLAAlign(kL1Size, 16U);
        uint32_t copyRowCnt = 0;
        for (uint32_t k1 = 0; k1 < kL1Loops; k1++) { // Tile K in L1 with a nested L0 loop.
            if (k1 == (kL1Loops - 1)) {
                // Tail block.
                kL1Size = kSize - (kL1Loops - 1) * K_L1_SPLIT_SIZE;
                kL1SizeAlign = SMLAAlign(kL1Size, 16U);
            }
            kvL1BufIter++;
            uint32_t kb = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
            bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
            uint32_t kOffset = k1 * kL0Loops;
            kL0Size = K_L0_SPLIT_SIZE;
            // Initialize kL0Size before computing kL0Loops; otherwise changes to kL0Size within the loop produce an incorrect loop count.
            kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
            kL0SizeAlign = kL0Size;
            for (uint32_t kL1 = kOffset; kL1 < kL0Loops + kOffset; kL1++) { // Copy PA data in chunks of 128.
                if (kL1 == kOffset + kL0Loops - 1) {
                    // Tail block.
                    kL0Size = kL1Size - (kL0Loops - 1) * kL0Size;
                    kL0SizeAlign = SMLAAlign(kL0Size, 16U);
                }

                uint32_t copyFinishRowCnt = 0;
                if (info.isOriOnly) {
                    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                        uint64_t curS2Offset = static_cast<uint64_t>(info.s2Idx) * constInfo.s2BaseSize +
                                               info.s2StartPoint + kL1 * K_L0_SPLIT_SIZE;
                        if (constInfo.hasOriSparseIndices) {
                            uint64_t mergeBase = static_cast<uint64_t>(info.cmpLoop % MERGE_CACHE_GM_BUF_NUM) *
                                                 N_WORKSPACE_SIZE * constInfo.headDim;
                            uint32_t blockElementCnt = 32U / sizeof(KV_T);
                            subvTensor = bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE];
                            for (uint32_t row = 0; row < kL0Size; ++row) {
                                uint64_t mergeOffset =
                                    mergeBase + static_cast<uint64_t>(kL1 * K_L0_SPLIT_SIZE + row) * constInfo.headDim +
                                    nL1 * N_SPLIT_SIZE;
                                GlobalTensor<KV_T> mergeSrcGm = kvMergeGm_[mergeOffset];
                                LocalTensor<KV_T> dstTensor = subvTensor[row * blockElementCnt];
                                DataCopyGmNDToL1<KV_T>(dstTensor, mergeSrcGm, 1, kL0SizeAlign, nL1Size,
                                                       constInfo.headDim);
                            }
                        } else
                            while (copyFinishRowCnt < kL0Size) {
                                copyRowCnt = constInfo.paOriBlockSize - curS2Offset % constInfo.paOriBlockSize;
                                if (copyFinishRowCnt + copyRowCnt > kL0Size) {
                                    copyRowCnt = kL0Size - copyFinishRowCnt;
                                }
                                Position startPos;
                                startPos.bIdx = info.bIdx;
                                startPos.n2Idx = info.n2Idx;
                                startPos.s2Idx = curS2Offset;
                                startPos.dIdx =
                                    nL1 * N_SPLIT_SIZE; // MM1 right matrix: BN2S2D, with D as the untiled K axis; MM2 right matrix: S2 is the K axis and D is tiled.
                                PAShape shape;
                                shape.blockSize = constInfo.paOriBlockSize;
                                shape.headNum = constInfo.kvHeadNum;
                                shape.headDim = constInfo.headDim;
                                shape.kvStride = constInfo.oriKvStride0;
                                shape.actHeadDim = nL1Size;
                                shape.maxblockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
                                shape.copyRowNum = copyRowCnt;
                                shape.copyRowNumAlign = kL0SizeAlign;
                                subvTensor =
                                    bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE + copyFinishRowCnt * 16];
                                DataCopyPA<KV_T>(subvTensor, oriKvGm, oriBlockTableGm, shape, startPos);

                                // Update loop variables.
                                copyFinishRowCnt += copyRowCnt;
                                curS2Offset += copyRowCnt;
                            }
                    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::BSND) {
                        subvTensor = bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE];
                        Nd2NzParams nd2nzPara;
                        nd2nzPara.ndNum = 1;
                        nd2nzPara.nValue = kL0Size; // Number of rows.
                        nd2nzPara.dValue = nL1Size; // 256
                        nd2nzPara.srcDValue = constInfo.headDim;
                        nd2nzPara.dstNzC0Stride = kL0SizeAlign;
                        nd2nzPara.dstNzNStride = 1;
                        nd2nzPara.srcNdMatrixStride = 0;
                        nd2nzPara.dstNzMatrixStride = 0;

                        uint32_t headStride = constInfo.headDim;
                        uint32_t seqStride = constInfo.kvHeadNum * constInfo.headDim;
                        uint64_t batchStride = (constInfo.oriKvStride0 == 0) ?
                                                   static_cast<uint64_t>(constInfo.kvSeqSize) * seqStride :
                                                   constInfo.oriKvStride0;

                        uint64_t curS2 = info.s2StartPoint + kL1 * K_L0_SPLIT_SIZE;
                        uint64_t offset = (uint64_t)info.bIdx * batchStride + (uint64_t)curS2 * seqStride +
                                          (uint64_t)info.n2Idx * headStride + nL1 * N_SPLIT_SIZE;
                        DataCopy(subvTensor, oriKvGm[offset], nd2nzPara);
                    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
                        uint64_t curS2Offset = static_cast<uint64_t>(info.s2Idx) * constInfo.s2BaseSize +
                                               info.s2StartPoint + kL1 * K_L0_SPLIT_SIZE;
                        Nd2NzParams nd2nzPara;
                        nd2nzPara.ndNum = 1;
                        nd2nzPara.nValue = kL0Size;      // Number of rows.
                        nd2nzPara.dValue = N_SPLIT_SIZE; // constInfo.headDim;
                        nd2nzPara.srcDValue = constInfo.headDim;
                        nd2nzPara.dstNzC0Stride = kL0SizeAlign;
                        nd2nzPara.dstNzNStride = 1;
                        nd2nzPara.srcNdMatrixStride = 0;
                        nd2nzPara.dstNzMatrixStride = 0;
                        DataCopy(bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE],
                                 oriKvGm[info.tensorBOffset + curS2Offset * constInfo.headDim + nL1 * N_SPLIT_SIZE],
                                 nd2nzPara);
                    }
                } else if (info.isOriCmpMix) {
                    uint32_t cumKl0LoopSize = kL1 * K_L0_SPLIT_SIZE;
                    uint32_t oriSizeCur = (info.actualSingleProcessSInnerOriSize < cumKl0LoopSize) ?
                                              0 :
                                              (info.actualSingleProcessSInnerOriSize - cumKl0LoopSize);
                    oriSizeCur = (oriSizeCur > kL0Size) ? kL0Size : oriSizeCur;
                    uint32_t cmpSizeCur = kL0Size - oriSizeCur;
                    cmpSizeCur = (cmpSizeCur < info.actualSingleProcessSInnerCmpSize) ?
                                     cmpSizeCur :
                                     info.actualSingleProcessSInnerCmpSize;
                    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                        if (oriSizeCur > 0) {
                            copyFinishRowCnt = 0;
                            uint64_t curS2Offset = info.s2StartPoint + kL1 * K_L0_SPLIT_SIZE;
                            while (copyFinishRowCnt < oriSizeCur) {
                                copyRowCnt = constInfo.paOriBlockSize - curS2Offset % constInfo.paOriBlockSize;
                                if (copyFinishRowCnt + copyRowCnt > oriSizeCur) {
                                    copyRowCnt = oriSizeCur - copyFinishRowCnt;
                                }
                                Position startPos;
                                startPos.bIdx = info.bIdx;
                                startPos.n2Idx = info.n2Idx;
                                startPos.s2Idx = curS2Offset;
                                startPos.dIdx =
                                    nL1 * N_SPLIT_SIZE; // MM1 right matrix: BN2S2D, with D as the untiled K axis; MM2 right matrix: S2 is the K axis and D is tiled.
                                PAShape shape;
                                shape.blockSize = constInfo.paOriBlockSize;
                                shape.headNum = constInfo.kvHeadNum;
                                shape.headDim = constInfo.headDim;
                                shape.kvStride = constInfo.oriKvStride0;
                                shape.actHeadDim = nL1Size;
                                shape.maxblockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
                                shape.copyRowNum = copyRowCnt;
                                shape.copyRowNumAlign = kL0SizeAlign;
                                subvTensor =
                                    bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE + copyFinishRowCnt * 16];
                                DataCopyPA<KV_T>(subvTensor, oriKvGm, oriBlockTableGm, shape, startPos);

                                // Update loop variables.
                                copyFinishRowCnt += copyRowCnt;
                                curS2Offset += copyRowCnt;
                            }
                        }
                        if (cmpSizeCur > 0) {
                            copyFinishRowCnt = 0;
                            uint32_t cmpMixsizeCur =
                                K_L0_SPLIT_SIZE - info.actualSingleProcessSInnerOriSize % K_L0_SPLIT_SIZE;
                            uint32_t oriOnlyLoopTimes = info.actualSingleProcessSInnerOriSize / K_L0_SPLIT_SIZE;
                            uint32_t cmpLoopTimes = kL1 - oriOnlyLoopTimes;
                            uint64_t curS2Offset =
                                (cmpLoopTimes == 0) ?
                                    0 :
                                    (cmpMixsizeCur + static_cast<uint64_t>(cmpLoopTimes - 1) * K_L0_SPLIT_SIZE);
                            while (copyFinishRowCnt < cmpSizeCur) {
                                copyRowCnt = constInfo.paCmpBlockSize - curS2Offset % constInfo.paCmpBlockSize;
                                if (copyFinishRowCnt + copyRowCnt > cmpSizeCur) {
                                    copyRowCnt = cmpSizeCur - copyFinishRowCnt;
                                }

                                Position startPos;
                                startPos.bIdx = info.bIdx;
                                startPos.n2Idx = info.n2Idx;
                                startPos.s2Idx = curS2Offset;
                                // TODO: Rename constants 256 and 32 for the seven-buffer scheme.
                                startPos.dIdx =
                                    nL1 * N_SPLIT_SIZE; // MM1 right matrix: BN2S2D, with D as the untiled K axis; MM2 right matrix: S2 is the K axis and D is tiled.

                                PAShape shape;
                                shape.blockSize = constInfo.paCmpBlockSize;
                                shape.headNum = constInfo.kvHeadNum;
                                shape.headDim = constInfo.headDim;
                                shape.kvStride = constInfo.cmpKvStride0;
                                shape.actHeadDim = nL1Size;
                                shape.maxblockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
                                shape.copyRowNum = copyRowCnt;
                                shape.copyRowNumAlign = kL0SizeAlign;
                                subvTensor = bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE +
                                                       oriSizeCur * 16 + copyFinishRowCnt * 16];
                                DataCopyPA<KV_T>(subvTensor, cmpKvGm, cmpBlockTableGm, shape, startPos);
                                // Update loop variables.
                                copyFinishRowCnt += copyRowCnt;
                                curS2Offset += copyRowCnt;
                            }
                        }
                    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::BSND) {
                        Nd2NzParams nd2nzPara;
                        nd2nzPara.ndNum = 1;
                        nd2nzPara.nValue = kL0Size; // Number of rows.
                        nd2nzPara.dValue = nL1Size; // 256
                        nd2nzPara.srcDValue = constInfo.headDim;
                        nd2nzPara.dstNzC0Stride = kL0SizeAlign;
                        nd2nzPara.dstNzNStride = 1;
                        nd2nzPara.srcNdMatrixStride = 0;
                        nd2nzPara.dstNzMatrixStride = 0;

                        uint32_t headStride = constInfo.headDim;
                        uint32_t seqStride = constInfo.kvHeadNum * constInfo.headDim;
                        if (oriSizeCur > 0) {
                            subvTensor = bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE];
                            nd2nzPara.nValue = oriSizeCur;
                            uint64_t batchStride = (constInfo.oriKvStride0 == 0) ?
                                                       static_cast<uint64_t>(constInfo.kvSeqSize) * seqStride :
                                                       constInfo.oriKvStride0;
                            uint64_t curS2 = info.s2StartPoint + kL1 * K_L0_SPLIT_SIZE;
                            uint64_t offset = (uint64_t)info.bIdx * batchStride + (uint64_t)curS2 * seqStride +
                                              (uint64_t)info.n2Idx * headStride + nL1 * N_SPLIT_SIZE;
                            DataCopy(subvTensor, oriKvGm[offset], nd2nzPara);
                        }
                        if (cmpSizeCur > 0) {
                            nd2nzPara.nValue = cmpSizeCur;
                            subvTensor = bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE + oriSizeCur * 16];
                            uint64_t batchStride = (constInfo.cmpKvStride0 == 0) ?
                                                       static_cast<uint64_t>(constInfo.cmpSeqSize) * seqStride :
                                                       constInfo.cmpKvStride0;
                            uint32_t cmpMixsizeCur =
                                K_L0_SPLIT_SIZE - info.actualSingleProcessSInnerOriSize % K_L0_SPLIT_SIZE;
                            uint32_t oriOnlyLoopTimes = info.actualSingleProcessSInnerOriSize / K_L0_SPLIT_SIZE;
                            uint32_t cmpLoopTimes = kL1 - oriOnlyLoopTimes;
                            uint64_t curS2 =
                                (cmpLoopTimes == 0) ?
                                    0 :
                                    (cmpMixsizeCur + static_cast<uint64_t>(cmpLoopTimes - 1) * K_L0_SPLIT_SIZE);
                            uint64_t offset = (uint64_t)info.bIdx * batchStride + (uint64_t)curS2 * seqStride +
                                              (uint64_t)info.n2Idx * headStride + nL1 * N_SPLIT_SIZE;
                            DataCopy(subvTensor, cmpKvGm[offset], nd2nzPara);
                        }
                    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
                        Nd2NzParams nd2nzPara;
                        nd2nzPara.ndNum = 1;
                        nd2nzPara.nValue = kL0Size;      // Number of rows.
                        nd2nzPara.dValue = N_SPLIT_SIZE; // constInfo.headDim;
                        nd2nzPara.srcDValue = constInfo.headDim;
                        nd2nzPara.dstNzC0Stride = kL0SizeAlign;
                        nd2nzPara.dstNzNStride = 1;
                        nd2nzPara.srcNdMatrixStride = 0;
                        nd2nzPara.dstNzMatrixStride = 0;

                        if (oriSizeCur > 0) {
                            nd2nzPara.nValue = oriSizeCur;
                            uint64_t curS2Offset = info.s2StartPoint + kL1 * K_L0_SPLIT_SIZE;
                            DataCopy(bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE],
                                     oriKvGm[info.tensorBOffset + curS2Offset * constInfo.headDim + nL1 * N_SPLIT_SIZE],
                                     nd2nzPara);
                        }
                        if (cmpSizeCur > 0) {
                            nd2nzPara.nValue = cmpSizeCur;
                            uint32_t cmpMixsizeCur =
                                K_L0_SPLIT_SIZE - info.actualSingleProcessSInnerOriSize % K_L0_SPLIT_SIZE;
                            uint32_t oriOnlyLoopTimes = info.actualSingleProcessSInnerOriSize / K_L0_SPLIT_SIZE;
                            uint32_t cmpLoopTimes = kL1 - oriOnlyLoopTimes;
                            uint64_t curS2Offset =
                                (cmpLoopTimes == 0) ?
                                    0 :
                                    (cmpMixsizeCur + static_cast<uint64_t>(cmpLoopTimes - 1) * K_L0_SPLIT_SIZE);
                            DataCopy(
                                bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE + oriSizeCur * 16],
                                cmpKvGm[info.tensorCmpBOffset + curS2Offset * constInfo.headDim + nL1 * N_SPLIT_SIZE],
                                nd2nzPara);
                        }
                    }
                } else {
                    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                        uint64_t curS2Offset = static_cast<uint64_t>(info.relativeS2Idx) * constInfo.s2BaseSize +
                                               info.s2StartPoint + K_L0_SPLIT_SIZE * kL1;
                        while (copyFinishRowCnt < kL0Size) {
                            copyRowCnt = constInfo.paCmpBlockSize - curS2Offset % constInfo.paCmpBlockSize;
                            if (copyFinishRowCnt + copyRowCnt > kL0Size) {
                                copyRowCnt = kL0Size - copyFinishRowCnt;
                            }

                            Position startPos;
                            startPos.bIdx = info.bIdx;
                            startPos.n2Idx = info.n2Idx;
                            startPos.s2Idx = curS2Offset;
                            // TODO: Rename constants 256 and 32 for the seven-buffer scheme.
                            startPos.dIdx =
                                nL1 * N_SPLIT_SIZE; // MM1 right matrix: BN2S2D, with D as the untiled K axis; MM2 right matrix: S2 is the K axis and D is tiled.

                            PAShape shape;
                            shape.blockSize = constInfo.paCmpBlockSize;
                            shape.headNum = constInfo.kvHeadNum;
                            shape.headDim = constInfo.headDim;
                            shape.kvStride = constInfo.cmpKvStride0;
                            shape.actHeadDim = nL1Size;
                            shape.maxblockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
                            shape.copyRowNum = copyRowCnt;
                            shape.copyRowNumAlign = kL0SizeAlign;
                            subvTensor =
                                bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE + copyFinishRowCnt * 16];
                            DataCopyPA<KV_T>(subvTensor, cmpKvGm, cmpBlockTableGm, shape, startPos);
                            // Update loop variables.
                            copyFinishRowCnt += copyRowCnt;
                            curS2Offset += copyRowCnt;
                        }
                    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::BSND) {
                        subvTensor = bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE];
                        Nd2NzParams nd2nzPara;
                        nd2nzPara.ndNum = 1;
                        nd2nzPara.nValue = kL0Size; // Number of rows.
                        nd2nzPara.dValue = nL1Size; // 256
                        nd2nzPara.srcDValue = constInfo.headDim;
                        nd2nzPara.dstNzC0Stride = kL0SizeAlign;
                        nd2nzPara.dstNzNStride = 1;
                        nd2nzPara.srcNdMatrixStride = 0;
                        nd2nzPara.dstNzMatrixStride = 0;

                        uint32_t headStride = constInfo.headDim;
                        uint32_t seqStride = constInfo.kvHeadNum * constInfo.headDim;
                        uint64_t batchStride = (constInfo.cmpKvStride0 == 0) ?
                                                   static_cast<uint64_t>(constInfo.cmpSeqSize) * seqStride :
                                                   constInfo.cmpKvStride0;

                        uint64_t curS2 = (uint64_t)info.relativeS2Idx * constInfo.s2BaseSize + info.s2StartPoint +
                                         K_L0_SPLIT_SIZE * kL1;
                        uint64_t offset = (uint64_t)info.bIdx * batchStride + (uint64_t)curS2 * seqStride +
                                          (uint64_t)info.n2Idx * headStride + nL1 * N_SPLIT_SIZE;
                        DataCopy(subvTensor, cmpKvGm[offset], nd2nzPara);
                    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
                        uint64_t curS2Offset = static_cast<uint64_t>(info.relativeS2Idx) * constInfo.s2BaseSize +
                                               info.s2StartPoint + K_L0_SPLIT_SIZE * kL1;
                        Nd2NzParams nd2nzPara;
                        nd2nzPara.ndNum = 1;
                        nd2nzPara.nValue = kL0Size;      // Number of rows.
                        nd2nzPara.dValue = N_SPLIT_SIZE; // constInfo.headDim;
                        nd2nzPara.srcDValue = constInfo.headDim;
                        nd2nzPara.dstNzC0Stride = kL0SizeAlign;
                        nd2nzPara.dstNzNStride = 1;
                        nd2nzPara.srcNdMatrixStride = 0;
                        nd2nzPara.dstNzMatrixStride = 0;
                        DataCopy(bL1Tensor[(kL1 - kOffset) * K_L0_SPLIT_SIZE * N_SPLIT_SIZE],
                                 cmpKvGm[info.tensorCmpBOffset + curS2Offset * constInfo.headDim + nL1 * N_SPLIT_SIZE],
                                 nd2nzPara);
                    }
                }
            }
            SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            mL1SizeAlign = M_SPLIT_SIZE;
            mL1Size = M_SPLIT_SIZE; // Actual M size.
            for (uint32_t mL1 = 0; mL1 < mL1Loops; mL1++) {
                if (mL1 == (mL1Loops - 1)) {
                    // Tail block.
                    mL1Size = mSize - (mL1Loops - 1) * M_SPLIT_SIZE;
                    mL1SizeAlign = SMLAAlign(mL1Size, 16U);
                }

                uint32_t mIdx = mBaseIdx + mL1;
                ka = GetQPL1RealIdx(mIdx, k1);
                LocalTensor<KV_T> aL1Tensor = l1QPTensor[ka * L1_BLOCK_OFFSET];
                if (nL1 == 0) {
                    WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]);
                    CopyInMm2AToL1(aL1Tensor, info, mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE, mL1Size, kL1Size,
                                   256 * k1);
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                }

                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(cL0BufIter % 2) *
                                      (L0C_PP_SIZE / sizeof(MM_OUT_T))]; // Keep cL0BufIter in step with M.
                uint32_t baseK = 128;
                uint32_t baseN = 128;
                kL0Size = 128;
                kL0SizeAlign = kL0Size;
                for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                    if (kL0 + 1 == kL0Loops) {
                        kL0Size = kL1Size - (kL0Loops - 1) * kL0Size;
                        kL0SizeAlign = SMLAAlign(kL0Size, 16U);
                    }
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadData3DParamsV2<KV_T> loadData3DParamsForB;
                    loadData3DParamsForB.l1H = kL0SizeAlign / 16; // Source operand height.
                    loadData3DParamsForB.l1W = 16;                // Source operand width is 16; destination height is l1H*l1W.
                    loadData3DParamsForB.padList[0] = 0;
                    loadData3DParamsForB.padList[1] = 0;
                    loadData3DParamsForB.padList[2] = 0;
                    loadData3DParamsForB.padList[3] = 255; // Tail data does not affect the sliding-window result.

                    loadData3DParamsForB.mExtension = kL0SizeAlign; // Transfer length along the destination height dimension.
                    loadData3DParamsForB.kExtension = nL1SizeAlign; // Transfer length along the destination width dimension.
                    loadData3DParamsForB.mStartPt = 0;              // Convolution kernel start along the destination width dimension.
                    loadData3DParamsForB.kStartPt = 0;              // Convolution kernel start along the destination height dimension.
                    loadData3DParamsForB.strideW = 1;
                    loadData3DParamsForB.strideH = 1;
                    loadData3DParamsForB.filterW = 1;
                    loadData3DParamsForB.filterSizeW = false; // Whether to extend the convolution kernel width by 256 elements beyond filterW.
                    loadData3DParamsForB.filterH = 1;
                    loadData3DParamsForB.filterSizeH = false; // Whether to extend the convolution kernel height by 256 elements beyond filterH.
                    loadData3DParamsForB.dilationFilterW = 1; // Convolution kernel width dilation factor.
                    loadData3DParamsForB.dilationFilterH = 1; // Convolution kernel height dilation factor.
                    loadData3DParamsForB.enTranspose = 1;     // Whether to enable transpose.
                    loadData3DParamsForB.fMatrixCtrl =
                        0; // Select FMATRIX_LEFT (0) or FMATRIX_RIGHT (1).
                    loadData3DParamsForB.channelSize =
                        nL1SizeAlign; // Source operand channel count. With dilation 1, destination width is filterW*filterH*channelSize.
                    LoadData<KV_T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[kL0 * baseK * baseN], loadData3DParamsForB);

                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadData3DParamsV2<KV_T> loadData3DParamsForA;
                    loadData3DParamsForA.l1H = mL1SizeAlign / 16; // Source operand height.
                    loadData3DParamsForA.l1W = 16;                // Source operand width.
                    loadData3DParamsForA.padList[0] = 0;
                    loadData3DParamsForA.padList[1] = 0;
                    loadData3DParamsForA.padList[2] = 0;
                    loadData3DParamsForA.padList[3] = 255; // Tail data does not affect the sliding-window result.

                    loadData3DParamsForA.mExtension = mL1SizeAlign; // Transfer length along the destination height dimension.
                    loadData3DParamsForA.kExtension = kL0SizeAlign; // Transfer length along the destination width dimension.
                    loadData3DParamsForA.mStartPt = 0;              // Convolution kernel start along the destination width dimension.
                    loadData3DParamsForA.kStartPt = 0;              // Convolution kernel start along the destination height dimension.
                    loadData3DParamsForA.strideW = 1;         // Convolution kernel stride along the source width dimension.
                    loadData3DParamsForA.strideH = 1;         // Convolution kernel stride along the source height dimension.
                    loadData3DParamsForA.filterW = 1;         // Convolution kernel width.
                    loadData3DParamsForA.filterSizeW = false; // Whether to extend the convolution kernel width by 256 elements beyond filterW.
                    loadData3DParamsForA.filterH = 1;         // Convolution kernel height.
                    loadData3DParamsForA.filterSizeH = false; // Whether to extend the convolution kernel height by 256 elements beyond filterH.
                    loadData3DParamsForA.dilationFilterW = 1; // Convolution kernel width dilation factor.
                    loadData3DParamsForA.dilationFilterH = 1; // Convolution kernel height dilation factor.
                    loadData3DParamsForA.enTranspose = 0; // Whether to transpose the entire destination matrix.
                    loadData3DParamsForA.fMatrixCtrl = 0;
                    loadData3DParamsForA.channelSize =
                        kL0SizeAlign; // Source operand channel count. With dilation 1, destination width is filterW*filterH*channelSize.
                    LoadData<KV_T, LOAD3DV2_CONFIG>(aL0Tensor, aL1Tensor[kL0 * baseK * mL1SizeAlign],
                                                    loadData3DParamsForA);
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));

                    MmadParams mmadParams;
                    mmadParams.m = mL1SizeAlign;
                    mmadParams.n = nL1SizeAlign;
                    mmadParams.k = kL0Size;
                    mmadParams.cmatrixInitVal = (kL0 == 0 && k1 == 0);
                    mmadParams.cmatrixSource = false;
                    mmadParams.unitFlag = ((k1 == (kL1Loops - 1)) && (kL0 == (kL0Loops - 1))) ? 0b11 : 0b10;

                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    abL0BufIter++;
                }

                if (nL1 == (nL1Loops - 1)) { // Final nL1 iteration: should B remain in L1 for the next computation?
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]); // Reverse synchronization: MTE1 has finished consuming A in L1.
                }

                if (k1 == (kL1Loops - 1)) {
                    // ND
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    fixParams.srcStride = mL1SizeAlign;
                    fixParams.dstStride = nSize; // Stride between adjacent mm2ResGm rows.
                    fixParams.ndNum = 1;         // Output in ND layout.
                    fixParams.unitFlag = 0b11;

                    uint64_t mm2Offset = (mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE) * nSize + nL1 * N_SPLIT_SIZE;
                    Fixpipe(mm2ResGm[(info.loop % (constInfo.preLoadNum)) * constInfo.bmm2ResUbSize + mm2Offset],
                            cL0Tensor, fixParams);
                }

                if (mL1Loops == 2) {
                    cL0BufIter++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]); // Reverse synchronization: MTE1 has finished consuming L1.
        }
        // cL0BufIter is no longer in use.
        if (mL1Loops == 1) {
            cL0BufIter++;
        }
    }
    qpL1BufIter += mL1Loops;
}
} // namespace SMLAKernel
#endif
