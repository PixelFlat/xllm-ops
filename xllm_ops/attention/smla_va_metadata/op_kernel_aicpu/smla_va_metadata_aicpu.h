/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file smla_va_metadata_aicpu.h
 * \brief
 */

#ifndef SMLA_VA_METADATA_AICPU_H
#define SMLA_VA_METADATA_AICPU_H

#include <array>
#include <string>
#include <vector>

#include "../../smla_va/op_kernel/smla_va_kernel_metadata.h"
#include "aicpu_common.h"
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"

namespace aicpu {
constexpr int64_t FA_TOLERANCE_RATIO = 2;
constexpr uint32_t COST_WEIGHT_M = 6U;
constexpr uint32_t COST_WEIGHT_S2 = 10U;
constexpr uint32_t BATCH_CONSISTENCY_MAX_REDUCTION_PARTS = 32U;
constexpr bool ORI_KV = false;
constexpr bool CMP_KV = true;
constexpr uint32_t NO_MASK = 0;
constexpr uint32_t HAS_MASK = 1;

enum BlockType : uint32_t {
  ORI_NORMAL_BLOCK = 0,
  ORI_TAIL_BLOCK,
  CMP_NORMAL_BLOCK,
  CMP_TAIL_BLOCK,
  BLOCK_MAX_TYPE
};

enum class SparseMode : uint8_t {
  DEFAULT_MASK = 0,
  ALL_MASK,
  LEFT_UP_CAUSAL,
  RIGHT_DOWN_CAUSAL,
  BAND,
  SPARSE_BUTT,
};

template <class T>
using Range = std::pair<T, T>;

template <class T>
using BlockCost = std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE)>,
                             static_cast<size_t>(BLOCK_MAX_TYPE)>;

template <typename T>
T Clip(T value, T minValue, T maxValue) {
  if (value < minValue) {
    return minValue;
  }
  if (value > maxValue) {
    return maxValue;
  }
  return value;
}

template <typename T>
inline bool IsWithinTolerance(T limit, T tolerance, T value) {
  return limit + tolerance >= value;
}

// Core-splitting output: FD metadata containing reduction data indices and core assignments.
struct FlashDecodeResult {
  uint32_t fdUsedVecNum{0U};  // Number of vector cores used for reduction.
  // 1. Index metadata for reduction tasks.
  std::vector<uint32_t>
      fdBN2Idx{};  // BN2 index per reduction task, indexed by task number, up to core count minus one.
  std::vector<uint32_t>
      fdMIdx{};  // GS1 index per reduction task, indexed by task number.
  std::vector<uint32_t>
      fdWorkspaceIdx{};  // Workspace location of each reduction task.
  std::vector<uint32_t>
      fdS2SplitNum{};  // Number of inter-core S2 splits per reduction task, indexed by task number.
  std::vector<uint32_t> fdMSize{};  // M-axis size per reduction task, indexed by task number.
  // 2. Vector-core assignments for reduction tasks during FD load balancing.
  std::vector<uint32_t>
      fdIdx{};  // Reduction task ID assigned to each vector core during FD load balancing.
  std::vector<uint32_t>
      fdMStart{};  // M-axis start of the reduction task assigned to each vector core during FD load balancing.
  std::vector<uint32_t>
      fdMNum{};  // M-axis row count of the reduction task assigned to each vector core during FD load balancing.

  FlashDecodeResult(uint32_t aicNum, uint32_t aivNum)
      : fdBN2Idx(aicNum),
        fdMIdx(aicNum),
        fdWorkspaceIdx(aicNum),
        fdS2SplitNum(aicNum),
        fdMSize(aicNum),
        fdIdx(aivNum),
        fdMStart(aivNum),
        fdMNum(aivNum) {}
};

// Core-splitting output: inter-core assignments for the FA stage.
struct SplitResult {
  uint32_t usedCoreNum{0U};        // Number of cores used.
  std::vector<uint32_t> bN2End{};  // BN2 end position of the data processed by each core.
  std::vector<uint32_t> gS1End{};  // GS1 end position of the data processed by each core.
  std::vector<uint32_t> s2End{};   // S2 end position of the data processed by each core.
  std::vector<uint32_t>
      firstFdDataWorkspaceIdx{};    // Storage location of the first reduction task on each core.
  int64_t maxCost{0};               // Cost of the slowest core.
  uint32_t numOfFdHead{0U};         // Number of reduction tasks.
  uint32_t maxS2SplitNum{0U};       // Maximum number of core splits for a reduction task.
  uint32_t maxS2LoopNum{0U};        // Maximum number of S2 iterations per core.
  FlashDecodeResult fdRes{0U, 0U};  // FD metadata.

  SplitResult(uint32_t aicNum, uint32_t aivNum)
      : bN2End(aicNum),
        gS1End(aicNum),
        s2End(aicNum),
        firstFdDataWorkspaceIdx(aicNum),
        fdRes(aicNum, aivNum) {};
};

// Internal core-splitting metadata: split information.
struct SplitInfo {
  std::vector<uint32_t> s1GBaseNum{};     // Number of basic blocks along S1G.
  std::vector<uint32_t> oriS2BaseNum{};   // Number of basic blocks along oriS2.
  std::vector<uint32_t> cmpS2BaseNum{};   // Number of basic blocks along cmpS2.
  std::vector<uint32_t> s1GTailSize{};    // Tail block size along S1G.
  std::vector<uint32_t> oriS2TailSize{};  // Tail block size along oriS2.
  std::vector<uint32_t> cmpS2TailSize{};  // Tail block size along cmpS2.
  bool isKvSeqAllZero{true};

  explicit SplitInfo(uint32_t batchSize)
      : s1GBaseNum(batchSize),
        oriS2BaseNum(batchSize),
        cmpS2BaseNum(batchSize),
        s1GTailSize(batchSize),
        oriS2TailSize(batchSize),
        cmpS2TailSize(batchSize) {}
};

// Internal core-splitting metadata: batch cost information.
struct CostInfo {
  std::vector<int64_t> bN2CostOfEachBatch{};     // Cost of the entire batch.
  std::vector<uint32_t> bN2BlockOfEachBatch{};   // Cost of the entire batch.
  std::vector<uint32_t> bN2S2LoopOfEachBatch{};  // Number of S2 iterations for the entire batch.
  std::vector<int64_t> bN2LastBlockCostOfEachBatch{};  // Cost of the last block in the batch.
  uint32_t totalBlockNum{0U};
  int64_t totalCost{0};
  int64_t maxS1GCost{0};  // Maximum cost across all S1G rows.

  explicit CostInfo(uint32_t batchSize)
      : bN2CostOfEachBatch(batchSize),
        bN2BlockOfEachBatch(batchSize),
        bN2S2LoopOfEachBatch(batchSize),
        bN2LastBlockCostOfEachBatch(batchSize) {}
};

// Internal core-splitting context: basic case metadata grouped to reduce interface arguments.
struct SplitContext {
  SplitInfo splitInfo{0U};
  CostInfo costInfo{0U};

  explicit SplitContext(uint32_t batchSize)
      : splitInfo(batchSize), costInfo(batchSize) {}
};

// Internal core-splitting context: temporary batch metadata.
struct BatchCache {
  uint32_t bIdx{0U};
  uint32_t s1Size{0U};
  uint32_t oriS2Size{0U};
  uint64_t cmpRevertS2Size{0U};
  int64_t oriPreTokenLeftUp{0};
  int64_t oriNextTokenLeftUp{0};
  int64_t cmpPreTokenLeftUp{0};
  int64_t cmpNextTokenLeftUp{0};
  BlockCost<int64_t> typeCost{};
};

// Internal core-splitting context: temporary metadata for the current S1G row.
struct S1GCache {
  uint32_t bIdx{0U};
  uint32_t s1GIdx{0U};
  uint32_t s2Start{0U};
  uint32_t s2End{0U};
  uint32_t oriS2Start{0U};
  uint32_t oriS2End{0U};
  uint32_t cmpS2Start{0U};  // Split point between the window and compressed parts.
  uint32_t cmpS2End{0U};
  int64_t s1GCost{0};
  int64_t s1GLastBlockCost{0};
  uint32_t s1GBlock{0U};
  uint32_t s2Loop{0U};
  uint32_t oriS1GBlock{0U};
  int64_t oriS1GCost{0};
  int64_t oriS1GLastBlockCost{0};
  int64_t oriS1GNormalBlockCost{0};
  uint32_t cmpS1GBlock{0U};
  int64_t cmpS1GCost{0};
  int64_t cmpS1GLastBlockCost{0};
  int64_t cmpS1GNormalBlockCost{0};
  int64_t oriS2TailSize{0};
  int64_t cmpS2TailSize{0};
  uint32_t actOriS2Size{0U};
  uint32_t actCmpS2Size{0U};
  uint32_t reductionBlockSize{0U};
};

// Internal core-splitting context: load on the current core during assignment.
struct CoreCache {
  int64_t costLimit{0};  // Load limit.
  int64_t cost{0};       // Assigned load.
  uint32_t block{0U};    // Number of assigned blocks.
  uint32_t s2Loop{0U};   // Number of assigned S2 iterations.
};

// Internal core-splitting context: assignment metadata.
struct AssignContext {
  uint32_t curBIdx{0U};
  uint32_t curBN2Idx{0U};
  uint32_t curS1GIdx{0U};
  uint32_t curS2Idx{0U};
  uint32_t curCoreIdx{0U};
  int64_t unassignedCost{0};
  uint32_t curKvSplitPart{1U};
  uint32_t preFdDataNum{0U};

  int64_t bN2Cost{0};
  uint32_t bN2Block{0U};
  uint32_t bN2S2Loop{0U};
  bool isFinished{false};
  BatchCache batchCache{};
  S1GCache s1GCache{};
  CoreCache coreCache{};
};

class SmlaVaMetadataCpuKernel : public CpuKernel {
 public:
  SmlaVaMetadataCpuKernel() = default;
  ~SmlaVaMetadataCpuKernel() = default;
  uint32_t Compute(CpuKernelContext& ctx) override;

 private:
  bool Prepare(CpuKernelContext& ctx);
  int32_t GetQueryBatchSize();
  void CalcOriMaskMode();
  void CalcCmpMaskMode();
  bool ParamsCheck();
  int32_t GetSumOfQuerySeq();
  bool ParamsInit();
  bool BalanceSchedule(SplitResult& splitRes);
  bool GenMetadata(SplitResult& splitRes);
  // util
  uint32_t GetS1SeqSize(uint32_t bIdx);
  uint32_t GetOriS2SeqSize(uint32_t bIdx);
  uint32_t GetCmpS2SeqSize(uint32_t bIdx);
  uint64_t GetRevertS2Size(uint32_t bIdx);
  uint32_t GetS1Idx(uint32_t s1Size, uint32_t s1GIdx);
  uint32_t GetBsStride(uint32_t bIdx, uint32_t s1Idx);
  uint32_t GetOriTopkLength(uint32_t bsStride);
  uint32_t GetOriTopkLength(uint32_t s1GIdx,
                            const BatchCache& batchCache) const;
  uint32_t ReadOriTopkLengthAtRow(uint32_t bIdx, uint32_t s1Idx) const;
  uint32_t GetCmpTopkLength(uint32_t bsStride);
  int64_t CalcOriPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
  int64_t CalcOriNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
  int64_t CalcCmpPreTokenLeftUp(uint32_t s1Size, uint64_t s2Size);
  int64_t CalcCmpNextTokenLeftUp(uint32_t s1Size, uint64_t s2Size);
  Range<int64_t> CalcS2TokenRange(uint32_t s1GIdx,
                                  const BatchCache& batchCache,
                                  bool isCmpKv);
  int64_t OriCalcCost(uint32_t basicM, uint32_t basicS2);
  int64_t CmpCalcCost(uint32_t basicM, uint32_t basicS2);
  void CalcCostTable(uint32_t s1GTailSize,
                     uint32_t reductionBlockSize,
                     uint32_t oriS2TailSize,
                     uint32_t cmpS2TailSize);

  // cache calculation
  void CalcBatchCache(uint32_t bIdx,
                      const SplitContext& splitContext,
                      BatchCache& batchCache);
  void CalcOriBlockRange(const Range<int64_t>& oriS2TokenRange,
                         const BatchCache& batchCache,
                         S1GCache& s1GCache);
  void CalcCmpBlockRange(const Range<int64_t>& cmpS2TokenRange,
                         const BatchCache& batchCache,
                         S1GCache& s1GCache);
  void CalcOriS1GCache(S1GCache& s1GCache, const SplitInfo& splitInfo);
  void CalcCmpS1GCache(S1GCache& s1GCache, const SplitInfo& splitInfo);
  void GatherOriAndCmpCache(S1GCache& s1GCache);
  void CalcS1GCache(uint32_t s1GIdx,
                    const SplitContext& splitContext,
                    const BatchCache& batchCache,
                    S1GCache& s1GCache);

  // preprocess
  void CalcSplitInfo(SplitContext& splitContext);
  void CalcBatchCost(uint32_t bIdx,
                     const SplitContext& splitContext,
                     CostInfo& costInfo);
  void CalcCostInfo(SplitContext& splitContext);

  // assign
  void UpdateCursor(const SplitContext& splitContext,
                    AssignContext& assignContext);
  void AssignByBatch(const SplitContext& splitContext,
                     AssignContext& assignContext);
  void AssignByRow(const SplitContext& splitContext,
                   AssignContext& assignContext);
  int64_t CalcCurBlockCost(const AssignContext& assignContext);
  uint32_t CalcCurBlockS2Loop(const AssignContext& assignContext);
  void AssignByBlock(const SplitContext& splitContext,
                     AssignContext& assignContext);
  void ForceAssign(const SplitContext& splitContext,
                   AssignContext& assignContext);
  void AssignBlocksToCore(const SplitContext& splitContext,
                          AssignContext& assignContext,
                          SplitResult& result);

  // FD
  bool IsNeedRecordFDInfo(const AssignContext& assignContext,
                          const SplitResult& splitRes);
  bool IsFirstReductionBlock(const AssignContext& assignContext,
                             const SplitResult& splitRes);
  void RecordFDInfo(const SplitContext& splitContext,
                    const AssignContext& assignContext,
                    SplitResult& result);

  // main
  void SplitFD(SplitResult& splitRes);
  void CalcSplitPlan(int64_t costLimit,
                     const SplitContext& splitContext,
                     SplitResult& result);

 private:
  // input
  Tensor* cuSeqlensQ_ = nullptr;
  Tensor* cuSeqlensOriKv_ = nullptr;
  Tensor* cuSeqlensCmpKv_ = nullptr;
  Tensor* sequsedQ_ = nullptr;
  Tensor* sequsedOriKv_ = nullptr;
  Tensor* sequsedCmpKv_ = nullptr;
  Tensor* cmpResidualKv_ = nullptr;
  Tensor* oriTopkLength_ = nullptr;
  Tensor* cmpTopkLength_ = nullptr;

  // output
  Tensor* metadata_ = nullptr;

  // attributes
  int32_t batchSize_ = 0;
  int32_t maxSeqlenQ_ = 0;
  int32_t numHeadsQ_ = 0;
  int32_t maxSeqlenOriKv_ = 0;
  int32_t maxSeqlenCmpKv_ = 0;
  int32_t numHeadsKv_ = 1;
  int32_t headDim_ = 0;
  int32_t oriTopK_ = 0;
  int32_t cmpTopK_ = 0;
  int32_t cmpRatio_ = 0;
  int32_t oriMaskMode_ = static_cast<int32_t>(SparseMode::BAND);
  int32_t cmpMaskMode_ = static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL);
  int64_t oriWinLeft_ = 127;
  int64_t oriWinRight_ = 0;
  std::string layoutQ_ = "BSND";
  std::string layoutKv_ = "PA_BBND";
  bool hasOriKv_ = true;
  bool hasCmpKv_ = true;
  uint32_t aicCoreNum_ = optiling::AIC_CORE_MAX_NUM;
  uint32_t aivCoreNum_ = optiling::AIV_CORE_MAX_NUM;
  bool isBatchConsistency_ = false;

  // attr
  std::string socVersion_ = "Ascend910_93";
  int64_t oriPreToken_ = 0;
  int64_t oriNextToken_ = 0;
  int64_t cmpPreToken_ = 0;
  int64_t cmpNextToken_ = 0;
  uint32_t groupSize_ = 0;
  uint32_t mBaseSize_ = 0;
  uint32_t s2BaseSize_ = 128U;
  bool isS1G_ = true;
  bool supportFd_ = false;
  uint32_t oriAttentionMode_ = HAS_MASK;
  uint32_t cmpAttentionMode_ = HAS_MASK;
  BlockCost<int64_t> typeCost_ = {};
  bool isSparseOriKv_ = false;
  bool isSparseCmpKv_ = false;
  bool hasOriTopkLength_ = false;
  uint32_t remainedBlockNum_ = 0U;

 private:
  enum class ParamId : uint32_t {
    // input
    cuSeqlensQ = 0,
    cuSeqlensOriKv = 1,
    cuSeqlensCmpKv = 2,
    sequsedQ = 3,
    sequsedOriKv = 4,
    sequsedCmpKv = 5,
    cmpResidualKv = 6,
    oriTopkLength = 7,
    cmpTopkLength = 8,
    // output
    metaData = 0,
  };
};
}  // namespace aicpu

#endif
