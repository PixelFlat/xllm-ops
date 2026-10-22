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
 * \file smla_va_metadata_aicpu.cpp
 * \brief
 */

#include "smla_va_metadata_aicpu.h"

#include <cmath>
#include <cstdio>

#include "log.h"
#include "status.h"

using namespace optiling;

namespace aicpu {
uint32_t SmlaVaMetadataCpuKernel::Compute(CpuKernelContext& ctx) {
  bool success = Prepare(ctx);
  if (!success) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  SplitResult splitRes{aicCoreNum_, aivCoreNum_};
  success = BalanceSchedule(splitRes) && GenMetadata(splitRes);
  return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool SmlaVaMetadataCpuKernel::Prepare(CpuKernelContext& ctx) {
  // input
  cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
  cuSeqlensOriKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensOriKv));
  cuSeqlensCmpKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensCmpKv));
  sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
  sequsedOriKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedOriKv));
  sequsedCmpKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedCmpKv));
  cmpResidualKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cmpResidualKv));
  oriTopkLength_ = ctx.Input(static_cast<uint32_t>(ParamId::oriTopkLength));
  cmpTopkLength_ = ctx.Input(static_cast<uint32_t>(ParamId::cmpTopkLength));
  hasOriTopkLength_ =
      (oriTopkLength_ != nullptr && oriTopkLength_->GetData() != nullptr &&
       oriTopkLength_->NumElements() > 0);
  // output
  metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

  bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                       GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
                       GetAttrValue(ctx, "head_dim", headDim_);
  if (!requiredAttrs) {
    return false;
  }

  // attributes optional
  GetAttrValueOpt(ctx, "soc_version", socVersion_);
  GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);
  GetAttrValueOpt(ctx, "aiv_core_num", aivCoreNum_);
  GetAttrValueOpt(ctx, "batch_size", batchSize_);
  GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
  GetAttrValueOpt(ctx, "max_seqlen_ori_kv", maxSeqlenOriKv_);
  GetAttrValueOpt(ctx, "max_seqlen_cmp_kv", maxSeqlenCmpKv_);
  GetAttrValueOpt(ctx, "ori_topk", oriTopK_);
  GetAttrValueOpt(ctx, "cmp_topk", cmpTopK_);
  GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);
  GetAttrValueOpt(ctx, "ori_mask_mode", oriMaskMode_);
  GetAttrValueOpt(ctx, "cmp_mask_mode", cmpMaskMode_);
  GetAttrValueOpt(ctx, "ori_win_left", oriWinLeft_);
  GetAttrValueOpt(ctx, "ori_win_right", oriWinRight_);
  GetAttrValueOpt(ctx, "layout_q", layoutQ_);
  GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
  GetAttrValueOpt(ctx, "has_ori_kv", hasOriKv_);
  GetAttrValueOpt(ctx, "has_cmp_kv", hasCmpKv_);
  GetAttrValueOpt(ctx, "is_batch_consistency", isBatchConsistency_);
  return (ParamsCheck() && ParamsInit());
}

bool SmlaVaMetadataCpuKernel::ParamsCheck() {
  // Validate that the output metadata is not null.
  if (metadata_ == nullptr) {
    KERNEL_LOG_ERROR("Output metadata is nullptr");
    return false;
  } else if (metadata_->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Output metadata data is nullptr");
    return false;
  }
  int32_t batchSize = GetQueryBatchSize();
  // Validate cu_seqlens_q elements.
  if (layoutQ_ == "TND") {
    if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
      const int32_t* cuSeqlensQPtr =
          static_cast<const int32_t*>(cuSeqlensQ_->GetData());
      // Validate that the first cu_seqlens_q element is zero.
      if (cuSeqlensQPtr[0] != 0) {
        KERNEL_LOG_ERROR(
            "The first element of cu_seqlens_q should be 0, but got %d",
            cuSeqlensQPtr[0]);
        return false;
      }
      for (int i = 0; i < batchSize + 1; i++) {
        // Validate that cu_seqlens_q elements are increasing.
        if (i > 0 && cuSeqlensQPtr[i - 1] > cuSeqlensQPtr[i]) {
          KERNEL_LOG_ERROR(
              "The elements in cu_seqlens_q must be in ascending order, "
              "but got cu_seqlens_q[%d] = %d, cu_seqlens_q[%d] = %d",
              i - 1,
              cuSeqlensQPtr[i - 1],
              i,
              cuSeqlensQPtr[i]);
          return false;
        }
      }
    }
  }
  // Validate seqused_q elements.
  if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
    const int32_t* sequsedQPtr =
        static_cast<const int32_t*>(sequsedQ_->GetData());
    const int32_t* cuSeqlensQPtr =
        (layoutQ_ == "TND" && cuSeqlensQ_ != nullptr &&
         cuSeqlensQ_->GetData() != nullptr)
            ? static_cast<const int32_t*>(cuSeqlensQ_->GetData())
            : nullptr;
    for (int i = 0; i < batchSize; i++) {
      // Validate that seqused_q elements are nonnegative.
      if (sequsedQPtr[i] < 0) {
        KERNEL_LOG_ERROR(
            "The elements in seqused_q should be >= 0, but got seqused_q[%d] = "
            "%d",
            i,
            sequsedQPtr[i]);
        return false;
      }
      // Validate that seqused_q elements do not exceed max_seqlen_q (BSND) or the cu_seqlens_q sequence length.
      // (TND)
      if (layoutQ_ == "BSND" && sequsedQPtr[i] > maxSeqlenQ_) {
        KERNEL_LOG_ERROR(
            "The elements in seqused_q should not be greater than max_seqlen_q "
            "%d, "
            "but got seqused_q[%d] = %d",
            maxSeqlenQ_,
            i,
            sequsedQPtr[i]);
        return false;
      }
      if (cuSeqlensQPtr != nullptr) {
        int32_t seqLen = cuSeqlensQPtr[i + 1] - cuSeqlensQPtr[i];
        if (sequsedQPtr[i] > seqLen) {
          KERNEL_LOG_ERROR(
              "The elements in seqused_q should not be greater than the "
              "sequence length "
              "from cu_seqlens_q %d, but got seqused_q[%d] = %d",
              seqLen,
              i,
              sequsedQPtr[i]);
          return false;
        }
      }
    }
  }
  if (hasOriKv_) {
    // Validate cu_seqlens_ori_kv elements.
    if (layoutKv_ == "TND") {
      if (cuSeqlensOriKv_ != nullptr && cuSeqlensOriKv_->GetData() != nullptr) {
        const int32_t* cuSeqlensOriKvPtr =
            static_cast<const int32_t*>(cuSeqlensOriKv_->GetData());
        // Validate that the first cu_seqlens_ori_kv element is zero.
        if (cuSeqlensOriKvPtr[0] != 0) {
          KERNEL_LOG_ERROR(
              "The first element of cu_seqlens_ori_kv should be 0, but got %d",
              cuSeqlensOriKvPtr[0]);
          return false;
        }
        for (int i = 0; i < batchSize + 1; i++) {
          // Validate that cu_seqlens_ori_kv elements are increasing.
          if (i > 0 && cuSeqlensOriKvPtr[i - 1] > cuSeqlensOriKvPtr[i]) {
            KERNEL_LOG_ERROR(
                "The elements in cu_seqlens_ori_kv must be in ascending order, "
                "but got cu_seqlens_ori_kv[%d] = %d, cu_seqlens_ori_kv[%d] = "
                "%d",
                i - 1,
                cuSeqlensOriKvPtr[i - 1],
                i,
                cuSeqlensOriKvPtr[i]);
            return false;
          }
        }
      }
    }
    // Validate seqused_ori_kv elements.
    if (sequsedOriKv_ != nullptr && sequsedOriKv_->GetData() != nullptr) {
      const int32_t* sequsedOriKvPtr =
          static_cast<const int32_t*>(sequsedOriKv_->GetData());
      const int32_t* cuSeqlensOriKvPtr =
          (layoutKv_ == "TND" && cuSeqlensOriKv_ != nullptr &&
           cuSeqlensOriKv_->GetData() != nullptr)
              ? static_cast<const int32_t*>(cuSeqlensOriKv_->GetData())
              : nullptr;
      for (int i = 0; i < batchSize; i++) {
        // Validate that seqused_ori_kv elements are nonnegative.
        if (sequsedOriKvPtr[i] < 0) {
          KERNEL_LOG_ERROR(
              "The elements in seqused_ori_kv should be >= 0, but got "
              "seqused_ori_kv[%d] = %d",
              i,
              sequsedOriKvPtr[i]);
          return false;
        }
        // Validate that seqused_ori_kv elements do not exceed max_seqlen_ori_kv (BSND) or
        // the cu_seqlens_ori_kv sequence length (TND).
        if (layoutKv_ == "BSND" && sequsedOriKvPtr[i] > maxSeqlenOriKv_) {
          KERNEL_LOG_ERROR(
              "The elements in seqused_ori_kv should not be greater than "
              "max_seqlen_ori_kv %d, but got seqused_ori_kv[%d] = %d",
              maxSeqlenOriKv_,
              i,
              sequsedOriKvPtr[i]);
          return false;
        }
        if (cuSeqlensOriKvPtr != nullptr) {
          int32_t seqLen = cuSeqlensOriKvPtr[i + 1] - cuSeqlensOriKvPtr[i];
          if (sequsedOriKvPtr[i] > seqLen) {
            KERNEL_LOG_ERROR(
                "The elements in seqused_ori_kv should not be greater than the "
                "sequence "
                "length from cu_seqlens_ori_kv %d, but got seqused_ori_kv[%d] "
                "= %d",
                seqLen,
                i,
                sequsedOriKvPtr[i]);
            return false;
          }
        }
      }
    }
    // Validate ori_topk_length elements.
    if (oriTopK_ != 0 &&
        oriMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK) &&
        oriTopkLength_ != nullptr && oriTopkLength_->GetData() != nullptr) {
      // Validate the number of ori_topk_length elements.
      int32_t sumOfQuerySeq = GetSumOfQuerySeq();
      const int32_t* oriTopkLengthPtr =
          static_cast<const int32_t*>(oriTopkLength_->GetData());
      auto oriTopkLengthShape = oriTopkLength_->GetTensorShape();
      int32_t oriTopkLengthSize = layoutQ_ == "TND"
                                      ? oriTopkLengthShape->GetDimSize(0) *
                                            oriTopkLengthShape->GetDimSize(1)
                                      : oriTopkLengthShape->GetDimSize(0) *
                                            oriTopkLengthShape->GetDimSize(1) *
                                            oriTopkLengthShape->GetDimSize(2);
      if (oriTopkLengthSize < sumOfQuerySeq) {
        KERNEL_LOG_ERROR(
            "The size of ori_topk_length %d should not be smaller than "
            "the sum of query sequence %d!",
            oriTopkLengthSize,
            sumOfQuerySeq);
        return false;
      }
      // Validate that ori_topk_length elements are nonnegative.
      for (int i = 0; i < oriTopkLengthSize; i++) {
        if (oriTopkLengthPtr[i] < 0) {
          KERNEL_LOG_ERROR(
              "The elements in ori_topk_length should be >= 0, but got "
              "ori_topk_length[%d] = %d",
              i,
              oriTopkLengthPtr[i]);
          return false;
        }
      }
    }
  }
  if (hasCmpKv_) {
    if (layoutKv_ == "TND") {
      // Validate cu_seqlens_cmp_kv elements.
      if (cuSeqlensCmpKv_ != nullptr && cuSeqlensCmpKv_->GetData() != nullptr) {
        const int32_t* cuSeqlensCmpKvPtr =
            static_cast<const int32_t*>(cuSeqlensCmpKv_->GetData());
        // Validate that the first cu_seqlens_cmp_kv element is zero.
        if (cuSeqlensCmpKvPtr[0] != 0) {
          KERNEL_LOG_ERROR(
              "The first element of cu_seqlens_cmp_kv should be 0, but got %d",
              cuSeqlensCmpKvPtr[0]);
          return false;
        }
        for (int i = 0; i < batchSize + 1; i++) {
          // Validate that cu_seqlens_cmp_kv elements are increasing.
          if (i > 0 && cuSeqlensCmpKvPtr[i - 1] > cuSeqlensCmpKvPtr[i]) {
            KERNEL_LOG_ERROR(
                "The elements in cu_seqlens_cmp_kv must be in ascending order, "
                "but got cu_seqlens_cmp_kv[%d] = %d, cu_seqlens_cmp_kv[%d] = "
                "%d",
                i - 1,
                cuSeqlensCmpKvPtr[i - 1],
                i,
                cuSeqlensCmpKvPtr[i]);
            return false;
          }
        }
      }
    }
    // Validate seqused_cmp_kv elements.
    if (sequsedCmpKv_ != nullptr && sequsedCmpKv_->GetData() != nullptr) {
      const int32_t* sequsedCmpKvPtr =
          static_cast<const int32_t*>(sequsedCmpKv_->GetData());
      const int32_t* cuSeqlensCmpKvPtr =
          (layoutKv_ == "TND" && cuSeqlensCmpKv_ != nullptr &&
           cuSeqlensCmpKv_->GetData() != nullptr)
              ? static_cast<const int32_t*>(cuSeqlensCmpKv_->GetData())
              : nullptr;
      for (int i = 0; i < batchSize; i++) {
        // Validate that seqused_cmp_kv elements are nonnegative.
        if (sequsedCmpKvPtr[i] < 0) {
          KERNEL_LOG_ERROR(
              "The elements in seqused_cmp_kv should be >= 0, but got "
              "seqused_cmp_kv[%d] = %d",
              i,
              sequsedCmpKvPtr[i]);
          return false;
        }
        // Validate that seqused_cmp_kv elements do not exceed max_seqlen_cmp_kv (BSND) or
        // the cu_seqlens_cmp_kv sequence length (TND).
        if (layoutKv_ == "BSND" && sequsedCmpKvPtr[i] > maxSeqlenCmpKv_) {
          KERNEL_LOG_ERROR(
              "The elements in seqused_cmp_kv should not be greater than "
              "max_seqlen_cmp_kv %d, but got seqused_cmp_kv[%d] = %d",
              maxSeqlenCmpKv_,
              i,
              sequsedCmpKvPtr[i]);
          return false;
        }
        if (cuSeqlensCmpKvPtr != nullptr) {
          int32_t seqLen = cuSeqlensCmpKvPtr[i + 1] - cuSeqlensCmpKvPtr[i];
          if (sequsedCmpKvPtr[i] > seqLen) {
            KERNEL_LOG_ERROR(
                "The elements in seqused_cmp_kv should not be greater than the "
                "sequence "
                "length from cu_seqlens_cmp_kv %d, but got seqused_cmp_kv[%d] "
                "= %d",
                seqLen,
                i,
                sequsedCmpKvPtr[i]);
            return false;
          }
        }
      }
    }
    // Validate cmp_residual_kv elements.
    if (cmpResidualKv_ != nullptr && cmpResidualKv_->GetData() != nullptr) {
      const int32_t* cmpResidualKvPtr =
          static_cast<const int32_t*>(cmpResidualKv_->GetData());
      for (int i = 0; i < batchSize; i++) {
        if (cmpResidualKvPtr[i] < 0 || cmpResidualKvPtr[i] >= cmpRatio_) {
          KERNEL_LOG_ERROR(
              "The elements in cmp_residual_kv should be in [0, "
              "cmpRatio_(%d)), but got "
              "cmp_residual_kv[%d] = %d",
              cmpRatio_,
              i,
              cmpResidualKvPtr[i]);
          return false;
        }
      }
    }
    // Validate cmp_topk_length elements.
    if (cmpTopK_ != 0 &&
        cmpMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK) &&
        cmpTopkLength_ != nullptr && cmpTopkLength_->GetData() != nullptr) {
      // Validate the number of cmp_topk_length elements.
      int32_t sumOfQuerySeq = GetSumOfQuerySeq();
      const int32_t* cmpTopkLengthPtr =
          static_cast<const int32_t*>(cmpTopkLength_->GetData());
      auto cmpTopkLengthShape = cmpTopkLength_->GetTensorShape();
      int32_t cmpTopkLengthSize = layoutQ_ == "TND"
                                      ? cmpTopkLengthShape->GetDimSize(0) *
                                            cmpTopkLengthShape->GetDimSize(1)
                                      : cmpTopkLengthShape->GetDimSize(0) *
                                            cmpTopkLengthShape->GetDimSize(1) *
                                            cmpTopkLengthShape->GetDimSize(2);
      if (cmpTopkLengthSize < sumOfQuerySeq) {
        KERNEL_LOG_ERROR(
            "The size of cmp_topk_length %d should not be smaller than "
            "the sum of query sequence %d!",
            cmpTopkLengthSize,
            sumOfQuerySeq);
        return false;
      }
      // Validate that cmp_topk_length elements are nonnegative.
      for (int i = 0; i < cmpTopkLengthSize; i++) {
        if (cmpTopkLengthPtr[i] < 0) {
          KERNEL_LOG_ERROR(
              "The elements in cmp_topk_length should be >= 0, but got "
              "cmp_topk_length[%d] = %d",
              i,
              cmpTopkLengthPtr[i]);
          return false;
        }
      }
    }
  }
  return true;
}

int32_t SmlaVaMetadataCpuKernel::GetSumOfQuerySeq() {
  int32_t batchSize = GetQueryBatchSize();
  // If sequsedQ_ is provided, use it to obtain BsSize.
  if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
    if (sequsedQ_->GetTensorShape() != nullptr) {
      const int32_t* seqUsedPtr =
          static_cast<const int32_t*>(sequsedQ_->GetData());
      int32_t queryBsSize = 0;
      for (int i = 0; i < batchSize; i++) {
        queryBsSize += seqUsedPtr[i];
      }
      return queryBsSize;
    }
  }
  // If sequsedQ_ is not provided, check the layout.
  if (layoutQ_ == "TND") {
    // For TND, try to obtain BsSize from cuSeqlensQ_.
    if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
      if (cuSeqlensQ_->GetTensorShape() != nullptr) {
        const int32_t* s1Ptr =
            static_cast<const int32_t*>(cuSeqlensQ_->GetData());
        return s1Ptr[batchSize];
      }
    }
  }
  // If the layout is not TND or cuSeqlensQ_ is empty, calculate BsSize from the shape.
  return batchSize_ * maxSeqlenQ_;
}

int32_t SmlaVaMetadataCpuKernel::GetQueryBatchSize() {
  // 1. If sequsedQ_ is provided, use it to obtain BatchSize.
  if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
    if (sequsedQ_->GetTensorShape() != nullptr) {
      return sequsedQ_->GetTensorShape()->GetDimSize(0);
    }
  }
  // 2. If sequsedQ_ is not provided, check the layout.
  if (layoutQ_ == "TND") {
    // For TND, try to obtain BatchSize from cuSeqlensQ_.
    if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
      if (cuSeqlensQ_->GetTensorShape() != nullptr) {
        return cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1;
      }
    }
  }
  // 3. If the layout is not TND or cuSeqlensQ_ is empty, use batchSize_.
  return batchSize_;
}

void SmlaVaMetadataCpuKernel::CalcOriMaskMode() {
  if (oriMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK)) {
    oriPreToken_ = INT64_MAX;
    oriNextToken_ = INT64_MAX;
    oriAttentionMode_ = NO_MASK;
  } else if (oriMaskMode_ ==
             static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
    oriPreToken_ = INT64_MAX;
    oriNextToken_ = 0;
    oriAttentionMode_ = HAS_MASK;
  } else {  // SparseMode = 4
    oriPreToken_ = (oriWinLeft_ > -1) ? oriWinLeft_ : INT64_MAX;
    oriNextToken_ = (oriWinRight_ > -1) ? oriWinRight_ : INT64_MAX;
    oriAttentionMode_ = HAS_MASK;
  }
}

void SmlaVaMetadataCpuKernel::CalcCmpMaskMode() {
  if (cmpMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK)) {
    cmpPreToken_ = INT64_MAX;
    cmpNextToken_ = INT64_MAX;
    cmpAttentionMode_ = NO_MASK;
  } else if (cmpMaskMode_ ==
             static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
    cmpPreToken_ = INT64_MAX;
    cmpNextToken_ = 0;
    cmpAttentionMode_ = HAS_MASK;
  } else {  // SparseMode = 4
    cmpPreToken_ = (oriWinLeft_ > -1) ? oriWinLeft_ : INT64_MAX;
    cmpNextToken_ = (oriWinRight_ > -1) ? oriWinRight_ : INT64_MAX;
    cmpAttentionMode_ = HAS_MASK;
  }
}

bool SmlaVaMetadataCpuKernel::ParamsInit() {
  batchSize_ = GetQueryBatchSize();
  CalcOriMaskMode();
  CalcCmpMaskMode();
  isS1G_ = (layoutQ_ == "BSND" || layoutQ_ == "BSH" || layoutQ_ == "TND");
  if (numHeadsKv_ == 0) {
    KERNEL_LOG_ERROR("num_heads_kv should not be 0.");
    return false;
  }
  groupSize_ = numHeadsQ_ / numHeadsKv_;
  if (hasOriKv_ && oriTopK_ != 0) {
    isSparseOriKv_ = true;
  }
  if (hasCmpKv_ && cmpTopK_ != 0) {
    isSparseCmpKv_ = true;
  }
  if (isBatchConsistency_) {
    supportFd_ = true;
  }

  if (isSparseOriKv_ && !isSparseCmpKv_) {
    mBaseSize_ = groupSize_;
  } else {
    mBaseSize_ = isSparseCmpKv_ ? groupSize_ : (256U / groupSize_) * groupSize_;
  }
  s2BaseSize_ = 512U;

  return true;
}

uint32_t SmlaVaMetadataCpuKernel::GetS1Idx(uint32_t s1Size, uint32_t s1GIdx) {
  uint32_t s1GToken = s1GIdx * mBaseSize_;
  uint32_t s1Idx = 0;
  if (isS1G_) {
    s1Idx = s1GToken / static_cast<int64_t>(groupSize_);
  } else {
    s1Idx = s1GToken % static_cast<int64_t>(s1Size);
  }
  return s1Idx;
}

uint32_t SmlaVaMetadataCpuKernel::GetBsStride(uint32_t bIdx, uint32_t s1Idx) {
  uint32_t bsStride = 0;
  if (layoutQ_ == "TND") {
    if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
      const int32_t* s1Ptr =
          static_cast<const int32_t*>(cuSeqlensQ_->GetData());
      bsStride = s1Ptr[bIdx] + s1Idx;
      return bsStride;
    }
  }
  bsStride = bIdx * static_cast<uint32_t>(maxSeqlenQ_) + s1Idx;
  return bsStride;
}

uint32_t SmlaVaMetadataCpuKernel::GetOriTopkLength(uint32_t bsStride) {
  // Try to use oriTopkLength_.
  if (oriTopK_ != 0 &&
      oriMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK) &&
      oriTopkLength_ != nullptr && oriTopkLength_->GetData() != nullptr) {
    const int32_t* oriTopkPtr =
        static_cast<const int32_t*>(oriTopkLength_->GetData());
    return static_cast<uint32_t>(oriTopkPtr[bsStride]);
  }
  // If the mask is not DEFAULT_MASK, use oriTopK_.
  return static_cast<uint32_t>(oriTopK_);
}

uint32_t SmlaVaMetadataCpuKernel::ReadOriTopkLengthAtRow(uint32_t bIdx,
                                                         uint32_t s1Idx) const {
  if (!hasOriTopkLength_) {
    return static_cast<uint32_t>(oriTopK_);
  }
  const int32_t* topkLenPtr =
      static_cast<const int32_t*>(oriTopkLength_->GetData());
  if (layoutQ_ == "TND") {
    uint32_t tIdx = s1Idx;
    if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
      const int32_t* cuPtr =
          static_cast<const int32_t*>(cuSeqlensQ_->GetData());
      tIdx = static_cast<uint32_t>(cuPtr[bIdx]) + s1Idx;
    } else {
      tIdx = bIdx * static_cast<uint32_t>(maxSeqlenQ_) + s1Idx;
    }
    return static_cast<uint32_t>(
        topkLenPtr[tIdx * static_cast<uint32_t>(numHeadsKv_)]);
  }
  return static_cast<uint32_t>(
      topkLenPtr[(static_cast<uint64_t>(bIdx) *
                      static_cast<uint64_t>(maxSeqlenQ_) +
                  s1Idx) *
                 static_cast<uint64_t>(numHeadsKv_)]);
}

uint32_t SmlaVaMetadataCpuKernel::GetOriTopkLength(
    uint32_t s1GIdx,
    const BatchCache& batchCache) const {
  if (!hasOriTopkLength_) {
    return static_cast<uint32_t>(oriTopK_);
  }
  int64_t s1GFirstToken =
      static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
  int64_t s1GLastToken =
      std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
               static_cast<int64_t>(batchCache.s1Size) *
                   static_cast<int64_t>(groupSize_)) -
      1;
  int64_t s1First = s1GFirstToken / static_cast<int64_t>(groupSize_);
  int64_t s1Last = s1GLastToken / static_cast<int64_t>(groupSize_);
  uint32_t maxLen = 0U;
  for (int64_t s1Idx = s1First; s1Idx <= s1Last; ++s1Idx) {
    maxLen = std::max(
        maxLen,
        ReadOriTopkLengthAtRow(batchCache.bIdx, static_cast<uint32_t>(s1Idx)));
  }
  return std::min(maxLen, static_cast<uint32_t>(oriTopK_));
}

uint32_t SmlaVaMetadataCpuKernel::GetCmpTopkLength(uint32_t bsStride) {
  // Try to use cmpTopkLength_.
  if (cmpTopK_ != 0 &&
      cmpMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK) &&
      cmpTopkLength_ != nullptr && cmpTopkLength_->GetData() != nullptr) {
    const int32_t* cmpTopkPtr =
        static_cast<const int32_t*>(cmpTopkLength_->GetData());
    return static_cast<uint32_t>(cmpTopkPtr[bsStride]);
  }
  // If the mask is not DEFAULT_MASK, use cmpTopK_.
  return static_cast<uint32_t>(cmpTopK_);
}

uint32_t SmlaVaMetadataCpuKernel::GetS1SeqSize(uint32_t bIdx) {
  // 1. If sequsedQ_ is provided, use it directly.
  if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
    const int32_t* seqUsedPtr =
        static_cast<const int32_t*>(sequsedQ_->GetData());
    return static_cast<uint32_t>(seqUsedPtr[bIdx]);
  }
  // 2. If sequsedQ_ is not provided, check the layout.
  if (layoutQ_ == "TND") {
    // For TND, try to use cuSeqlensQ_.
    if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
      const int32_t* s1Ptr =
          static_cast<const int32_t*>(cuSeqlensQ_->GetData());
      return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
    }
  }
  // 3. If the layout is not TND or cuSeqlensQ_ is empty, use maxSeqlenQ_.
  return static_cast<uint32_t>(maxSeqlenQ_);
}

uint32_t SmlaVaMetadataCpuKernel::GetOriS2SeqSize(uint32_t bIdx) {
  // If sequsedOriKv_ is provided, use it directly.
  if (sequsedOriKv_ != nullptr && sequsedOriKv_->GetData() != nullptr) {
    const int32_t* seqUsedPtr =
        static_cast<const int32_t*>(sequsedOriKv_->GetData());
    return static_cast<uint32_t>(seqUsedPtr[bIdx]);
  }
  // If sequsedOriKv_ is not provided, check the layout.
  if (layoutKv_ == "TND") {
    // For TND, try to use cuSeqlensOriKv_.
    if (cuSeqlensOriKv_ != nullptr && cuSeqlensOriKv_->GetData() != nullptr) {
      const int32_t* s2Ptr =
          static_cast<const int32_t*>(cuSeqlensOriKv_->GetData());
      return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
    }
  }
  // For PA, or if max_seqlen_ori_kv is not provided and ori_kv is sparse, try to obtain the length
  // from topk.
  if ((layoutKv_ == "PA_BBND" || maxSeqlenOriKv_ == 0) && isSparseOriKv_) {
    return UINT32_MAX;
  }
  // Use max_seqlen_ori_kv.
  return static_cast<uint32_t>(maxSeqlenOriKv_);
}

uint32_t SmlaVaMetadataCpuKernel::GetCmpS2SeqSize(uint32_t bIdx) {
  // If sequsedCmpKv_ is provided, use it directly.
  if (sequsedCmpKv_ != nullptr && sequsedCmpKv_->GetData() != nullptr) {
    const int32_t* seqUsedPtr =
        static_cast<const int32_t*>(sequsedCmpKv_->GetData());
    return static_cast<uint32_t>(seqUsedPtr[bIdx]);
  }
  // If sequsedCmpKv_ is not provided, check the layout.
  if (layoutKv_ == "TND") {
    // For TND, try to use cuSeqlensCmpKv_.
    if (cuSeqlensCmpKv_ != nullptr && cuSeqlensCmpKv_->GetData() != nullptr) {
      const int32_t* s2Ptr =
          static_cast<const int32_t*>(cuSeqlensCmpKv_->GetData());
      return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
    }
  }
  // For PA, or if max_seqlen_cmp_kv is not provided and cmp_kv is sparse,
  // try to obtain the length from topk.
  if ((layoutKv_ == "PA_BBND" || maxSeqlenCmpKv_ == 0) && isSparseCmpKv_) {
    return UINT32_MAX;
  }
  // Use max_seqlen_cmp_kv.
  return static_cast<uint32_t>(maxSeqlenCmpKv_);
}

uint64_t SmlaVaMetadataCpuKernel::GetRevertS2Size(uint32_t bIdx) {
  uint32_t cmpS2Size = GetCmpS2SeqSize(bIdx);
  if (cmpResidualKv_ != nullptr && cmpResidualKv_->GetData() != nullptr) {
    const int32_t* residualPtr =
        static_cast<const int32_t*>(cmpResidualKv_->GetData());
    return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_) +
           residualPtr[bIdx];
  } else {
    return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_);
  }
}

void SmlaVaMetadataCpuKernel::CalcSplitInfo(SplitContext& splitContext) {
  // Calculate splits for each batch, identify empty batches, and record the last valid batch. N2 splits are identical within each batch.
  SplitInfo& splitInfo = splitContext.splitInfo;
  for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
    uint32_t s1Size = GetS1SeqSize(bIdx);
    splitInfo.s1GBaseNum[bIdx] =
        (static_cast<uint64_t>(s1Size) * groupSize_ + (mBaseSize_ - 1U)) /
        mBaseSize_;
    splitInfo.s1GTailSize[bIdx] =
        (static_cast<uint64_t>(s1Size) * groupSize_) % mBaseSize_;
    if (hasOriKv_) {
      uint32_t curOriS2Size = GetOriS2SeqSize(bIdx);
      splitInfo.oriS2BaseNum[bIdx] =
          (static_cast<uint64_t>(curOriS2Size) + s2BaseSize_ - 1U) /
          s2BaseSize_;
    }
    if (hasCmpKv_) {
      uint32_t curCmpS2Size = GetCmpS2SeqSize(bIdx);
      splitInfo.cmpS2BaseNum[bIdx] =
          (static_cast<uint64_t>(curCmpS2Size) + s2BaseSize_ - 1U) /
          s2BaseSize_;
    }
    if (splitInfo.s1GBaseNum[bIdx] != 0U &&
        (splitInfo.oriS2BaseNum[bIdx] != 0U ||
         splitInfo.cmpS2BaseNum[bIdx] != 0U)) {
      splitInfo.isKvSeqAllZero = false;
    }
  }
}

int64_t SmlaVaMetadataCpuKernel::CalcOriPreTokenLeftUp(uint32_t s1Size,
                                                       uint32_t s2Size) {
  auto mode = static_cast<SparseMode>(oriMaskMode_);
  if (mode == SparseMode::BAND) {
    return oriPreToken_ == INT64_MAX
               ? INT64_MAX
               : static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) +
                     oriPreToken_;
  }
  return oriPreToken_;
}

int64_t SmlaVaMetadataCpuKernel::CalcOriNextTokenLeftUp(uint32_t s1Size,
                                                        uint32_t s2Size) {
  auto mode = static_cast<SparseMode>(oriMaskMode_);
  switch (mode) {
    case SparseMode::DEFAULT_MASK:
    case SparseMode::ALL_MASK:
    case SparseMode::LEFT_UP_CAUSAL:
      return oriNextToken_;
    case SparseMode::RIGHT_DOWN_CAUSAL:
      return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
    case SparseMode::BAND:
      return oriNextToken_ == INT64_MAX
                 ? INT64_MAX
                 : static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) +
                       oriNextToken_;
    default:
      return oriNextToken_;
  }
}

int64_t SmlaVaMetadataCpuKernel::CalcCmpPreTokenLeftUp(uint32_t s1Size,
                                                       uint64_t s2Size) {
  auto mode = static_cast<SparseMode>(cmpMaskMode_);
  if (mode == SparseMode::BAND) {
    return cmpPreToken_ == INT64_MAX
               ? INT64_MAX
               : static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) +
                     cmpPreToken_;
  }
  return cmpPreToken_;
}

int64_t SmlaVaMetadataCpuKernel::CalcCmpNextTokenLeftUp(uint32_t s1Size,
                                                        uint64_t s2Size) {
  auto mode = static_cast<SparseMode>(cmpMaskMode_);
  switch (mode) {
    case SparseMode::DEFAULT_MASK:
    case SparseMode::ALL_MASK:
    case SparseMode::LEFT_UP_CAUSAL:
      return cmpNextToken_;
    case SparseMode::RIGHT_DOWN_CAUSAL:
      return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
    case SparseMode::BAND:
      return cmpNextToken_ == INT64_MAX
                 ? INT64_MAX
                 : static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) +
                       cmpNextToken_;
    default:
      return cmpNextToken_;
  }
}

int64_t SmlaVaMetadataCpuKernel::OriCalcCost(uint32_t basicM,
                                             uint32_t basicS2) {
  uint32_t oriAlignCoefM = 16U;
  uint32_t oriAlignCoefS2 = 64U;
  uint32_t oriAlignBasicM = (basicM + oriAlignCoefM - 1U) / oriAlignCoefM;
  uint32_t oriAlignBasicS2 = (basicS2 + oriAlignCoefS2 - 1U) / oriAlignCoefS2;
  return static_cast<int64_t>(COST_WEIGHT_M * oriAlignBasicM +
                              COST_WEIGHT_S2 * oriAlignBasicS2);
}

int64_t SmlaVaMetadataCpuKernel::CmpCalcCost(uint32_t basicM,
                                             uint32_t basicS2) {
  uint32_t cmpAlignCoefM = 16U;
  uint32_t cmpAlignCoefS2 = 64U;
  uint32_t cmpAlignBasicM = (basicM + cmpAlignCoefM - 1U) / cmpAlignCoefM;
  uint32_t cmpAlignBasicS2 = (basicS2 + cmpAlignCoefS2 - 1U) / cmpAlignCoefS2;
  return static_cast<int64_t>(COST_WEIGHT_M * cmpAlignBasicM +
                              COST_WEIGHT_S2 * cmpAlignBasicS2);
}

void SmlaVaMetadataCpuKernel::CalcCostTable(uint32_t s1GTailSize,
                                            uint32_t reductionBlockSize,
                                            uint32_t oriS2TailSize,
                                            uint32_t cmpS2TailSize) {
  uint32_t normalS2Size = isBatchConsistency_ && reductionBlockSize > 0U
                              ? reductionBlockSize
                              : s2BaseSize_;
  // Cost of the original part.
  if (hasOriKv_) {
    typeCost_[ORI_NORMAL_BLOCK][ORI_NORMAL_BLOCK] =
        OriCalcCost(mBaseSize_, normalS2Size);
    typeCost_[ORI_TAIL_BLOCK][ORI_NORMAL_BLOCK] =
        (s1GTailSize == 0U) ? 0U : OriCalcCost(s1GTailSize, normalS2Size);
    typeCost_[ORI_NORMAL_BLOCK][ORI_TAIL_BLOCK] =
        (oriS2TailSize == 0U) ? 0U : OriCalcCost(mBaseSize_, oriS2TailSize);
    typeCost_[ORI_TAIL_BLOCK][ORI_TAIL_BLOCK] =
        (s1GTailSize == 0U || oriS2TailSize == 0U)
            ? 0U
            : OriCalcCost(s1GTailSize, oriS2TailSize);
  }
  // Cost of the compressed part.
  if (hasCmpKv_) {
    typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] =
        CmpCalcCost(mBaseSize_, normalS2Size);
    typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] =
        (s1GTailSize == 0U) ? 0U : CmpCalcCost(s1GTailSize, normalS2Size);
    typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] =
        (cmpS2TailSize == 0U) ? 0U : CmpCalcCost(mBaseSize_, cmpS2TailSize);
    typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] =
        (s1GTailSize == 0U || cmpS2TailSize == 0U)
            ? 0U
            : CmpCalcCost(s1GTailSize, cmpS2TailSize);
  }
}

Range<int64_t> SmlaVaMetadataCpuKernel::CalcS2TokenRange(
    uint32_t s1GIdx,
    const BatchCache& batchCache,
    bool isCmpKv) {
  // actual seq == 0
  if (!isCmpKv) {
    if (batchCache.s1Size == 0U || batchCache.oriS2Size == 0U) {
      return std::make_pair(0, 0);
    }
    if (isSparseOriKv_ && !hasCmpKv_) {
      uint32_t oriTopkSize = GetOriTopkLength(s1GIdx, batchCache);
      if (oriTopkSize == 0U) {
        return std::make_pair(0, 0);
      }
      return std::make_pair(0, static_cast<int64_t>(oriTopkSize) - 1);
    }
  } else {
    if (batchCache.s1Size == 0U || batchCache.cmpRevertS2Size == 0U) {
      return std::make_pair(0, 0);
    }
  }

  // no mask
  uint32_t hasMask = 1;
  int64_t s2Size = isCmpKv ? static_cast<int64_t>(batchCache.cmpRevertS2Size)
                           : static_cast<int64_t>(batchCache.oriS2Size);
  hasMask = isCmpKv ? cmpAttentionMode_ : oriAttentionMode_;
  if (!hasMask) {
    return std::make_pair(0, s2Size - 1);
  }

  // 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken,
  // s1GLastToken
  int64_t s1GFirstToken =
      static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
  int64_t s1GLastToken =
      std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
               static_cast<int64_t>(batchCache.s1Size) *
                   static_cast<int64_t>(groupSize_)) -
      1;

  int64_t s1FirstToken = 0;
  int64_t s1LastToken = 0;
  if (isS1G_) {
    s1FirstToken = s1GFirstToken / static_cast<int64_t>(groupSize_);
    s1LastToken = s1GLastToken / static_cast<int64_t>(groupSize_);
  } else {
    if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
      // start and end locate in one G
      s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
      s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
    } else {
      // start and end locate in tow or more G, but working same as crossing a
      // complete block
      s1FirstToken = 0;
      s1LastToken = batchCache.s1Size;
    }
  }

  int64_t s2FirstToken = 0;
  int64_t s2LastToken = 0;
  if (!isCmpKv) {
    s2FirstToken = s1FirstToken - batchCache.oriPreTokenLeftUp;
    s2LastToken = batchCache.oriNextTokenLeftUp == INT64_MAX
                      ? INT64_MAX
                      : s1LastToken + batchCache.oriNextTokenLeftUp;
  } else {
    s2FirstToken = s1FirstToken - batchCache.cmpPreTokenLeftUp;
    s2LastToken = batchCache.cmpNextTokenLeftUp == INT64_MAX
                      ? INT64_MAX
                      : s1LastToken + batchCache.cmpNextTokenLeftUp;
  }
  return std::make_pair(s2FirstToken, s2LastToken);
}

void SmlaVaMetadataCpuKernel::CalcBatchCache(uint32_t bIdx,
                                             const SplitContext& splitContext,
                                             BatchCache& batchCache) {
  const SplitInfo& splitInfo = splitContext.splitInfo;

  batchCache.bIdx = bIdx;
  batchCache.s1Size = GetS1SeqSize(bIdx);
  if (hasOriKv_) {
    batchCache.oriS2Size = GetOriS2SeqSize(bIdx);
    batchCache.oriPreTokenLeftUp =
        CalcOriPreTokenLeftUp(batchCache.s1Size, batchCache.oriS2Size);
    batchCache.oriNextTokenLeftUp =
        CalcOriNextTokenLeftUp(batchCache.s1Size, batchCache.oriS2Size);
  }
  if (hasCmpKv_) {
    batchCache.cmpRevertS2Size = GetRevertS2Size(bIdx);
    batchCache.cmpPreTokenLeftUp =
        CalcCmpPreTokenLeftUp(batchCache.s1Size, batchCache.cmpRevertS2Size);
    batchCache.cmpNextTokenLeftUp =
        CalcCmpNextTokenLeftUp(batchCache.s1Size, batchCache.cmpRevertS2Size);
  }
}

void SmlaVaMetadataCpuKernel::CalcOriS1GCache(S1GCache& s1GCache,
                                              const SplitInfo& splitInfo) {
  // Process block metadata for the original part.
  if (s1GCache.oriS2Start >= s1GCache.oriS2End) {
    // An invalid original range makes the entire S1G row effectively empty.
    s1GCache.oriS1GBlock = 0;
    s1GCache.oriS1GCost = 0;
    s1GCache.oriS1GLastBlockCost = 0;
    s1GCache.oriS1GNormalBlockCost = 0;
  } else {
    // Calculate the block count and cost along the original axis.
    s1GCache.oriS1GBlock = s1GCache.oriS2End - s1GCache.oriS2Start;
    // Check whether the original S2 axis has a tail block.
    uint32_t curOriTailS2Num = (s1GCache.oriS2TailSize != 0U) ? 1U : 0U;
    uint32_t curOriNormalS2Num = s1GCache.oriS1GBlock - curOriTailS2Num;
    if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) &&
        splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
      s1GCache.oriS1GCost =
          typeCost_[ORI_TAIL_BLOCK][ORI_NORMAL_BLOCK] * curOriNormalS2Num +
          typeCost_[ORI_TAIL_BLOCK][ORI_TAIL_BLOCK] * curOriTailS2Num;
      s1GCache.oriS1GLastBlockCost =
          curOriTailS2Num > 0U ? typeCost_[ORI_TAIL_BLOCK][ORI_TAIL_BLOCK]
                               : typeCost_[ORI_TAIL_BLOCK][ORI_NORMAL_BLOCK];
      s1GCache.oriS1GNormalBlockCost =
          typeCost_[ORI_TAIL_BLOCK][ORI_NORMAL_BLOCK];
    } else {
      s1GCache.oriS1GCost =
          typeCost_[ORI_NORMAL_BLOCK][ORI_NORMAL_BLOCK] * curOriNormalS2Num +
          typeCost_[ORI_NORMAL_BLOCK][ORI_TAIL_BLOCK] * curOriTailS2Num;
      s1GCache.oriS1GLastBlockCost =
          curOriTailS2Num > 0U ? typeCost_[ORI_NORMAL_BLOCK][ORI_TAIL_BLOCK]
                               : typeCost_[ORI_NORMAL_BLOCK][ORI_NORMAL_BLOCK];
      s1GCache.oriS1GNormalBlockCost =
          typeCost_[ORI_NORMAL_BLOCK][ORI_NORMAL_BLOCK];
    }
  }
}

void SmlaVaMetadataCpuKernel::CalcCmpS1GCache(S1GCache& s1GCache,
                                              const SplitInfo& splitInfo) {
  // Process block metadata for the compressed part.
  if (s1GCache.cmpS2Start >= s1GCache.cmpS2End) {
    // If the compressed range is invalid, keep the cost at zero.
    s1GCache.cmpS1GBlock = 0;
    s1GCache.cmpS1GCost = 0;
    s1GCache.cmpS1GLastBlockCost = 0;
    s1GCache.cmpS1GNormalBlockCost = 0;
  } else {
    // Calculate the block count and cost along the compressed axis.
    s1GCache.cmpS1GBlock = s1GCache.cmpS2End - s1GCache.cmpS2Start;
    // Check whether the compressed S2 axis has a tail block.
    uint32_t curCmpTailS2Num = (s1GCache.cmpS2TailSize != 0U)
                                   ? 1U
                                   : 0U;  // Updated check using local var
    uint32_t curCmpNormalS2Num = s1GCache.cmpS1GBlock - curCmpTailS2Num;
    if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) &&
        splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
      s1GCache.cmpS1GCost =
          typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] * curCmpNormalS2Num +
          typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] * curCmpTailS2Num;
      s1GCache.cmpS1GLastBlockCost =
          curCmpTailS2Num > 0U ? typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK]
                               : typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK];
      s1GCache.cmpS1GNormalBlockCost =
          typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK];
    } else {
      s1GCache.cmpS1GCost =
          typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] * curCmpNormalS2Num +
          typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] * curCmpTailS2Num;
      s1GCache.cmpS1GLastBlockCost =
          curCmpTailS2Num > 0U ? typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK]
                               : typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK];
      s1GCache.cmpS1GNormalBlockCost =
          typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK];
    }
  }
}

void SmlaVaMetadataCpuKernel::CalcOriBlockRange(
    const Range<int64_t>& oriS2TokenRange,
    const BatchCache& batchCache,
    S1GCache& s1GCache) {
  int64_t oriS2FirstToken = oriS2TokenRange.first;
  int64_t oriS2LastToken = oriS2TokenRange.second;
  s1GCache.oriS2Start = 0;
  // S2 start, end, and tail size for the original part.
  if (oriS2FirstToken >= static_cast<int64_t>(batchCache.oriS2Size) ||
      oriS2LastToken < 0 || oriS2LastToken < oriS2FirstToken) {
    s1GCache.oriS2End = 0;
    s1GCache.oriS2TailSize = 0;
  } else {
    oriS2FirstToken = Clip(oriS2FirstToken,
                           static_cast<int64_t>(0),
                           static_cast<int64_t>(batchCache.oriS2Size - 1U));
    oriS2LastToken = Clip(oriS2LastToken,
                          static_cast<int64_t>(0),
                          static_cast<int64_t>(batchCache.oriS2Size - 1U));
    // Take the minimum of oriS2LastToken and topk.
    uint32_t s1Idx = GetS1Idx(batchCache.s1Size, s1GCache.s1GIdx);
    uint32_t bsStride = GetBsStride(s1GCache.bIdx, s1Idx);
    uint32_t oriTopkSize =
        isSparseOriKv_
            ? GetOriTopkLength(s1GCache.s1GIdx, batchCache)
            : GetOriTopkLength(GetBsStride(
                  s1GCache.bIdx, GetS1Idx(batchCache.s1Size, s1GCache.s1GIdx)));
    s1GCache.actOriS2Size =
        isSparseOriKv_
            ? std::min(
                  static_cast<uint32_t>(oriS2LastToken - oriS2FirstToken + 1),
                  oriTopkSize)
            : static_cast<uint32_t>(oriS2LastToken - oriS2FirstToken + 1);
    s1GCache.oriS2End = s1GCache.actOriS2Size == 0
                            ? 0
                            : (s1GCache.actOriS2Size - 1U) / s2BaseSize_ + 1U;
    s1GCache.oriS2TailSize = s1GCache.actOriS2Size % s2BaseSize_;
  }
}

void SmlaVaMetadataCpuKernel::CalcCmpBlockRange(
    const Range<int64_t>& cmpRevertS2TokenRange,
    const BatchCache& batchCache,
    S1GCache& s1GCache) {
  int64_t cmpRevertS2FirstToken = cmpRevertS2TokenRange.first;
  int64_t cmpRevertS2LastToken = cmpRevertS2TokenRange.second;
  s1GCache.cmpS2Start = s1GCache.oriS2End;
  // S2 start, end, and tail size for the compressed part.
  if (cmpRevertS2FirstToken >=
          static_cast<int64_t>(batchCache.cmpRevertS2Size) ||
      cmpRevertS2LastToken < 0 ||
      cmpRevertS2LastToken < cmpRevertS2FirstToken) {
    s1GCache.cmpS2End = s1GCache.cmpS2Start;
    s1GCache.cmpS2TailSize = 0;
  } else {
    cmpRevertS2FirstToken =
        Clip(cmpRevertS2FirstToken,
             static_cast<int64_t>(0),
             static_cast<int64_t>(batchCache.cmpRevertS2Size - 1U));
    cmpRevertS2LastToken =
        Clip(cmpRevertS2LastToken,
             static_cast<int64_t>(0),
             static_cast<int64_t>(batchCache.cmpRevertS2Size - 1U));
    // Return immediately if the compressed length is zero.
    if ((cmpRevertS2LastToken + 1) / cmpRatio_ == 0) {
      s1GCache.cmpS2End = s1GCache.cmpS2Start;
      s1GCache.cmpS2TailSize = 0;
      return;
    }
    // Get the compressed token index.
    uint64_t cmpS2FirstToken =
        (cmpRevertS2FirstToken + 1) / cmpRatio_ == 0
            ? 0
            : (cmpRevertS2FirstToken + 1) / cmpRatio_ - 1U;
    uint64_t cmpS2LastToken = (cmpRevertS2LastToken + 1) / cmpRatio_ - 1U;
    // Take the minimum of cmpS2LastToken and topk.
    uint32_t s1Idx = GetS1Idx(batchCache.s1Size, s1GCache.s1GIdx);
    uint32_t bsStride = GetBsStride(s1GCache.bIdx, s1Idx);
    uint32_t cmpTopkSize = GetCmpTopkLength(bsStride);
    s1GCache.actCmpS2Size =
        isSparseCmpKv_
            ? std::min(
                  static_cast<uint32_t>(cmpS2LastToken - cmpS2FirstToken + 1),
                  cmpTopkSize)
            : static_cast<uint32_t>(cmpS2LastToken - cmpS2FirstToken + 1);
    s1GCache.cmpS2End = s1GCache.actCmpS2Size == 0
                            ? s1GCache.cmpS2Start
                            : s1GCache.cmpS2Start +
                                  (s1GCache.actCmpS2Size - 1U) / s2BaseSize_ +
                                  1U;
    s1GCache.cmpS2TailSize = s1GCache.actCmpS2Size % s2BaseSize_;
  }
}

void SmlaVaMetadataCpuKernel::GatherOriAndCmpCache(S1GCache& s1GCache) {
  s1GCache.s2Start = 0;
  if (s1GCache.cmpS1GBlock > 0) {
    s1GCache.s1GLastBlockCost = s1GCache.cmpS1GLastBlockCost;
    s1GCache.s2End = s1GCache.cmpS2End;
  } else {
    s1GCache.s1GLastBlockCost = s1GCache.oriS1GLastBlockCost;
    s1GCache.s2End = s1GCache.oriS2End;
  }
  s1GCache.s1GBlock = s1GCache.oriS1GBlock + s1GCache.cmpS1GBlock;
  s1GCache.s1GCost = s1GCache.oriS1GCost + s1GCache.cmpS1GCost;
}

void SmlaVaMetadataCpuKernel::CalcS1GCache(uint32_t s1GIdx,
                                           const SplitContext& splitContext,
                                           const BatchCache& batchCache,
                                           S1GCache& s1GCache) {
  const SplitInfo& splitInfo = splitContext.splitInfo;
  // Return immediately if the S1G row is empty.
  if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
    s1GCache.s1GCost = 0;
    s1GCache.s1GLastBlockCost = 0;
    s1GCache.oriS1GNormalBlockCost = 0;
    s1GCache.oriS1GLastBlockCost = 0;
    s1GCache.cmpS1GNormalBlockCost = 0;
    s1GCache.cmpS1GLastBlockCost = 0;
    s1GCache.s1GBlock = 0;
    s1GCache.s2Loop = 0;
    s1GCache.s2Start = 0;
    s1GCache.cmpS2Start = 0;
    s1GCache.s2End = 0;
    return;
  }
  s1GCache.bIdx = batchCache.bIdx;
  s1GCache.s1GIdx = s1GIdx;
  s1GCache.actOriS2Size = 0U;
  s1GCache.actCmpS2Size = 0U;
  s1GCache.reductionBlockSize = 0U;
  // Calculate the effective workload boundaries for ori_kv.
  if (hasOriKv_) {
    // Calculate the ori_kv S2 token boundaries.
    auto oriS2TokenRange = CalcS2TokenRange(s1GIdx, batchCache, ORI_KV);
    // Calculate the ori_kv S2 block boundaries.
    CalcOriBlockRange(oriS2TokenRange, batchCache, s1GCache);
  } else {
    // Initialize the ori_kv S2 token boundaries to zero.
    s1GCache.oriS2Start = 0;
    s1GCache.oriS2End = s1GCache.oriS2Start;
    s1GCache.oriS2TailSize = 0;
  }
  // Calculate the effective workload boundaries for cmp_kv.
  if (hasCmpKv_) {
    // Calculate the cmp_kv S2 token boundaries.
    auto cmpRevertS2TokenRange = CalcS2TokenRange(s1GIdx, batchCache, CMP_KV);
    // Calculate the cmp_kv S2 block boundaries.
    CalcCmpBlockRange(cmpRevertS2TokenRange, batchCache, s1GCache);
  } else {
    // Initialize the cmp_kv S2 token boundaries to zero.
    s1GCache.cmpS2Start = s1GCache.oriS2End;
    s1GCache.cmpS2End = s1GCache.cmpS2Start;
    s1GCache.cmpS2TailSize = 0;
  }
  if (isBatchConsistency_) {
    // The reduction block only depends on this row, so its reduction tree is
    // independent of the surrounding batch.
    uint64_t actTotalS2Size =
        static_cast<uint64_t>(s1GCache.actOriS2Size) + s1GCache.actCmpS2Size;
    if (actTotalS2Size > 0U) {
      uint64_t rawReductionBlockSize =
          actTotalS2Size / BATCH_CONSISTENCY_MAX_REDUCTION_PARTS;
      s1GCache.reductionBlockSize =
          static_cast<uint32_t>((rawReductionBlockSize + s2BaseSize_ - 1U) /
                                s2BaseSize_ * s2BaseSize_);
      s1GCache.reductionBlockSize =
          std::max(s1GCache.reductionBlockSize, s2BaseSize_);
      s1GCache.oriS2End =
          s1GCache.actOriS2Size == 0U
              ? 0U
              : (s1GCache.actOriS2Size - 1U) / s1GCache.reductionBlockSize + 1U;
      s1GCache.oriS2TailSize =
          s1GCache.actOriS2Size % s1GCache.reductionBlockSize;
      s1GCache.cmpS2Start = s1GCache.oriS2End;
      s1GCache.cmpS2End =
          s1GCache.actCmpS2Size == 0U
              ? s1GCache.cmpS2Start
              : s1GCache.cmpS2Start +
                    (s1GCache.actCmpS2Size - 1U) / s1GCache.reductionBlockSize +
                    1U;
      s1GCache.cmpS2TailSize =
          s1GCache.actCmpS2Size % s1GCache.reductionBlockSize;
    }
  }
  // Calculate the workload of a basic block.
  CalcCostTable(splitInfo.s1GTailSize[s1GCache.bIdx],
                s1GCache.reductionBlockSize,
                s1GCache.oriS2TailSize,
                s1GCache.cmpS2TailSize);
  // Calculate cost and block metadata for the original and compressed parts.
  CalcOriS1GCache(s1GCache, splitInfo);
  CalcCmpS1GCache(s1GCache, splitInfo);
  // Aggregate cost and block metadata for the original and compressed parts.
  GatherOriAndCmpCache(s1GCache);
  s1GCache.s2Loop = static_cast<uint32_t>(
      (static_cast<uint64_t>(s1GCache.actOriS2Size) + s2BaseSize_ - 1U) /
          s2BaseSize_ +
      (static_cast<uint64_t>(s1GCache.actCmpS2Size) + s2BaseSize_ - 1U) /
          s2BaseSize_);
}

void SmlaVaMetadataCpuKernel::CalcBatchCost(uint32_t bIdx,
                                            const SplitContext& splitContext,
                                            CostInfo& costInfo) {
  const SplitInfo& splitInfo = splitContext.splitInfo;

  costInfo.bN2CostOfEachBatch[bIdx] = 0;
  costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
  costInfo.bN2S2LoopOfEachBatch[bIdx] = 0U;
  costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

  if (GetS1SeqSize(bIdx) == 0U) {
    return;
  }
  if (!hasOriKv_ && !hasCmpKv_) {
    return;
  } else if (!hasOriKv_) {
    if (GetCmpS2SeqSize(bIdx) == 0U) {
      return;
    }
  } else if (!hasCmpKv_) {
    if (GetOriS2SeqSize(bIdx) == 0U) {
      return;
    }
  } else {
    if ((GetOriS2SeqSize(bIdx) == 0U) && GetCmpS2SeqSize(bIdx) == 0U) {
      return;
    }
  }

  BatchCache bCache;
  S1GCache s1GCache;
  CalcBatchCache(bIdx, splitContext, bCache);
  for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
    CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
    costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
    costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;
    costInfo.bN2S2LoopOfEachBatch[bIdx] += s1GCache.s2Loop;
    // Update the maximum S1G row cost.
    if (s1GCache.s1GCost > costInfo.maxS1GCost) {
      costInfo.maxS1GCost = s1GCache.s1GCost;
    }
    if (s1GCache.s1GBlock > 0) {
      costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
    }
  }
}

void SmlaVaMetadataCpuKernel::CalcCostInfo(SplitContext& splitContext) {
  const SplitInfo& splitInfo = splitContext.splitInfo;
  CostInfo& costInfo = splitContext.costInfo;

  if (splitInfo.isKvSeqAllZero) {
    costInfo.totalCost = 0;
    costInfo.totalBlockNum = 0U;
    return;
  }

  // Calculate and record the batch workload for batch-based assignment.
  // Calculate row boundaries and aggregate block counts and workload.
  for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
    CalcBatchCost(bIdx, splitContext, costInfo);
    costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * numHeadsKv_;
    costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * numHeadsKv_;
  }
}

void SmlaVaMetadataCpuKernel::UpdateCursor(const SplitContext& splitContext,
                                           AssignContext& assignContext) {
  const SplitInfo& splitInfo = splitContext.splitInfo;
  const CostInfo& costInfo = splitContext.costInfo;

  bool UpdateS1G = false;
  bool UpdateBatch = false;

  // Update S2
  if (assignContext.curS2Idx >=
      assignContext.s1GCache.s2End) {  // assignInfo.s2End is an exclusive boundary.
    assignContext.curS2Idx = 0U;
    assignContext.curS1GIdx++;
    UpdateS1G = true;
  }

  // Update S1G
  if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
    assignContext.curS1GIdx = 0U;
    assignContext.curBN2Idx++;
  }

  // Update Batch
  if (assignContext.curBN2Idx ==
      batchSize_ *
          numHeadsKv_) {  // All workload has been assigned; set the exclusive end boundary for the last core and return.
    assignContext.curS1GIdx = 0U;
    assignContext.curS2Idx = 0U;
    assignContext.isFinished = true;
    return;
  }

  if (assignContext.curBN2Idx / numHeadsKv_ != assignContext.curBIdx) {
    assignContext.curBIdx = assignContext.curBN2Idx / numHeadsKv_;
    assignContext.curS1GIdx = 0U;
    UpdateBatch = true;
    UpdateS1G = true;
  }

  // Update Cache
  if (UpdateBatch) {
    CalcBatchCache(
        assignContext.curBIdx, splitContext, assignContext.batchCache);
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block =
        costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    assignContext.bN2S2Loop =
        costInfo.bN2S2LoopOfEachBatch[assignContext.curBIdx];
  }
  if (UpdateS1G) {
    CalcS1GCache(assignContext.curS1GIdx,
                 splitContext,
                 assignContext.batchCache,
                 assignContext.s1GCache);
    assignContext.curS2Idx =
        supportFd_ ? assignContext.s1GCache.oriS2Start : 0U;
  }
}

void SmlaVaMetadataCpuKernel::AssignByBatch(const SplitContext& splitContext,
                                            AssignContext& assignContext) {
  if (assignContext.isFinished) {
    return;
  }
  const CostInfo& costInfo = splitContext.costInfo;
  const SplitInfo& splitInfo = splitContext.splitInfo;
  while (assignContext.bN2Cost == 0 ||
         IsWithinTolerance(
             assignContext.coreCache.costLimit,
             costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] /
                 FA_TOLERANCE_RATIO,
             assignContext.coreCache.cost + assignContext.bN2Cost)) {
    assignContext.coreCache.cost += assignContext.bN2Cost;
    assignContext.coreCache.block += assignContext.bN2Block;
    assignContext.coreCache.s2Loop += assignContext.bN2S2Loop;
    assignContext.curBN2Idx++;
    // to the end
    if (assignContext.curBN2Idx == batchSize_ * numHeadsKv_) {
      assignContext.curS1GIdx = 0U;
      assignContext.curS2Idx = 0U;
      assignContext.isFinished = true;
      return;
    }

    // next batch
    if (assignContext.curBN2Idx / numHeadsKv_ != assignContext.curBIdx) {
      assignContext.curBIdx = assignContext.curBN2Idx / numHeadsKv_;
      CalcBatchCache(
          assignContext.curBIdx, splitContext, assignContext.batchCache);
    }

    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block =
        costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    assignContext.bN2S2Loop =
        costInfo.bN2S2LoopOfEachBatch[assignContext.curBIdx];
    assignContext.curS1GIdx = 0U;
    CalcS1GCache(assignContext.curS1GIdx,
                 splitContext,
                 assignContext.batchCache,
                 assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;
  }
}

void SmlaVaMetadataCpuKernel::AssignByRow(const SplitContext& splitContext,
                                          AssignContext& assignContext) {
  if (assignContext.isFinished) {
    return;
  }

  while (IsWithinTolerance(
      assignContext.coreCache.costLimit,
      assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
      assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
    assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
    assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;
    assignContext.coreCache.s2Loop += assignContext.s1GCache.s2Loop;
    // A row has been assigned from the current batch; update the remaining workload.
    assignContext.bN2Cost =
        assignContext.bN2Cost > assignContext.s1GCache.s1GCost
            ? assignContext.bN2Cost - assignContext.s1GCache.s1GCost
            : 0;
    assignContext.bN2Block =
        assignContext.bN2Block > assignContext.s1GCache.s1GBlock
            ? assignContext.bN2Block - assignContext.s1GCache.s1GBlock
            : 0U;
    assignContext.bN2S2Loop =
        assignContext.bN2S2Loop > assignContext.s1GCache.s2Loop
            ? assignContext.bN2S2Loop - assignContext.s1GCache.s2Loop
            : 0U;
    // Calculate metadata for the new row.
    do {
      assignContext.curS1GIdx++;
      CalcS1GCache(assignContext.curS1GIdx,
                   splitContext,
                   assignContext.batchCache,
                   assignContext.s1GCache);
    } while (assignContext.s1GCache.s1GBlock == 0);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;
  }
}

int64_t SmlaVaMetadataCpuKernel::CalcCurBlockCost(
    const AssignContext& assignContext) {
  int64_t curCost = 0;
  if (assignContext.curS2Idx < assignContext.s1GCache.cmpS2Start) {
    curCost = assignContext.s1GCache.oriS1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.cmpS2Start - 1U)) {
      curCost = assignContext.s1GCache.oriS1GLastBlockCost;
    }
  } else {
    curCost = assignContext.s1GCache.cmpS1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
      curCost = assignContext.s1GCache.cmpS1GLastBlockCost;
    }
  }
  return curCost;
}

uint32_t SmlaVaMetadataCpuKernel::CalcCurBlockS2Loop(
    const AssignContext& assignContext) {
  if (!isBatchConsistency_) {
    return 1U;
  }
  const S1GCache& s1GCache = assignContext.s1GCache;
  uint32_t blockSize = s1GCache.reductionBlockSize;
  if (assignContext.curS2Idx < s1GCache.cmpS2Start) {
    if (assignContext.curS2Idx + 1U == s1GCache.cmpS2Start &&
        s1GCache.oriS2TailSize != 0) {
      blockSize = static_cast<uint32_t>(s1GCache.oriS2TailSize);
    }
  } else if (assignContext.curS2Idx + 1U == s1GCache.s2End &&
             s1GCache.cmpS2TailSize != 0) {
    blockSize = static_cast<uint32_t>(s1GCache.cmpS2TailSize);
  }
  return (blockSize + s2BaseSize_ - 1U) / s2BaseSize_;
}

void SmlaVaMetadataCpuKernel::AssignByBlock(const SplitContext& splitContext,
                                            AssignContext& assignContext) {
  if (assignContext.isFinished || !supportFd_) {
    return;
  }

  int64_t curCost = CalcCurBlockCost(assignContext);
  uint32_t curS2Loop = CalcCurBlockS2Loop(assignContext);

  // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost; assign at least one block.
  while (IsWithinTolerance(assignContext.coreCache.costLimit,
                           curCost / FA_TOLERANCE_RATIO,
                           assignContext.coreCache.cost + curCost)) {
    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.coreCache.s2Loop += curS2Loop;
    assignContext.curS2Idx++;
    // A block has been assigned from the current batch; update the remaining workload.
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    // A block has been assigned from the current row; update the remaining workload.
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.bN2Block--;
    assignContext.s1GCache.s1GBlock--;
    assignContext.bN2S2Loop = assignContext.bN2S2Loop > curS2Loop
                                  ? assignContext.bN2S2Loop - curS2Loop
                                  : 0U;
    assignContext.s1GCache.s2Loop =
        assignContext.s1GCache.s2Loop > curS2Loop
            ? assignContext.s1GCache.s2Loop - curS2Loop
            : 0U;
    curCost = CalcCurBlockCost(assignContext);
    curS2Loop = CalcCurBlockS2Loop(assignContext);
  }
}

void SmlaVaMetadataCpuKernel::ForceAssign(const SplitContext& splitContext,
                                          AssignContext& assignContext) {
  if (assignContext.isFinished) {
    return;
  }

  int64_t curCost = CalcCurBlockCost(assignContext);
  uint32_t curS2Loop = CalcCurBlockS2Loop(assignContext);

  assignContext.coreCache.cost += curCost;
  assignContext.coreCache.block++;
  assignContext.coreCache.s2Loop += curS2Loop;
  assignContext.curS2Idx++;
  // A block has been assigned from the current batch; update the remaining workload.
  assignContext.bN2Cost = assignContext.bN2Cost - curCost;
  assignContext.bN2Block--;
  assignContext.bN2S2Loop = assignContext.bN2S2Loop > curS2Loop
                                ? assignContext.bN2S2Loop - curS2Loop
                                : 0U;
  // A block has been assigned from the current row; update the remaining workload.
  assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
  assignContext.s1GCache.s1GBlock--;
  assignContext.s1GCache.s2Loop =
      assignContext.s1GCache.s2Loop > curS2Loop
          ? assignContext.s1GCache.s2Loop - curS2Loop
          : 0U;
  UpdateCursor(splitContext, assignContext);
}

bool SmlaVaMetadataCpuKernel::IsNeedRecordFDInfo(
    const AssignContext& assignContext,
    const SplitResult& splitRes) {
  // Split points rarely coincide with row ends. Defer reduction bookkeeping until the next split point determines whether reduction is needed.
  // Core zero requires no processing.
  if (assignContext.curCoreIdx == 0U) {
    return false;
  }
  // No row spans multiple cores; no processing is needed.
  if (assignContext.curKvSplitPart <= 1U) {
    return false;
  }
  // The row requiring reduction has not been fully processed.
  if (assignContext.curBN2Idx ==
          splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
      assignContext.curS1GIdx ==
          splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
    return false;
  }
  return true;
}

bool SmlaVaMetadataCpuKernel::IsFirstReductionBlock(
    const AssignContext& assignContext,
    const SplitResult& splitRes) {
  if (assignContext.curCoreIdx == 0U ||
      splitRes.s2End[assignContext.curCoreIdx - 1U] == 0U) {
    return true;
  }
  return assignContext.curBN2Idx !=
             splitRes.bN2End[assignContext.curCoreIdx - 1U] ||
         assignContext.curS1GIdx !=
             splitRes.gS1End[assignContext.curCoreIdx - 1U];
}

void SmlaVaMetadataCpuKernel::RecordFDInfo(const SplitContext& splitContext,
                                           const AssignContext& assignContext,
                                           SplitResult& result) {
  const SplitInfo& splitInfo = splitContext.splitInfo;
  // The row requiring reduction contains the split point of the previous core.
  uint32_t splitBIdx =
      result.bN2End[assignContext.curCoreIdx - 1U] / numHeadsKv_;
  uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
  uint32_t s1Size = GetS1SeqSize(splitBIdx);

  // Calculate balanced FD assignments for the reduction data.
  uint32_t curFdS1gSize =
      (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U)
          ? (static_cast<uint64_t>(s1Size) * groupSize_ -
             static_cast<uint64_t>(splitS1GIdx) * mBaseSize_)
          : mBaseSize_;
  // Record the reduction metadata.
  result.maxS2SplitNum =
      std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
  // If a head reduction exists, its split point must be the end position of the previous core.
  result.fdRes.fdBN2Idx[result.numOfFdHead] =
      result.bN2End[assignContext.curCoreIdx - 1U];
  result.fdRes.fdMIdx[result.numOfFdHead] =
      result.gS1End[assignContext.curCoreIdx - 1U];
  result.fdRes.fdWorkspaceIdx[result.numOfFdHead] = assignContext.preFdDataNum;
  result.fdRes.fdS2SplitNum[result.numOfFdHead] = assignContext.curKvSplitPart;
  result.fdRes.fdMSize[result.numOfFdHead] = curFdS1gSize;
  result.numOfFdHead++;
}

void SmlaVaMetadataCpuKernel::AssignBlocksToCore(
    const SplitContext& splitContext,
    AssignContext& assignContext,
    SplitResult& result) {
  const CostInfo& costInfo = splitContext.costInfo;
  result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] =
      assignContext.preFdDataNum + assignContext.curKvSplitPart - 1U;
  int64_t avgCost =
      assignContext.unassignedCost / (aicCoreNum_ - assignContext.curCoreIdx);
  assignContext.coreCache = {};
  if (!supportFd_) {
    assignContext.coreCache.costLimit = std::max(avgCost, costInfo.maxS1GCost);
  } else {
    assignContext.coreCache.costLimit = avgCost;
  }
  // 1. Assign whole batches.
  AssignByBatch(splitContext, assignContext);
  // 2. Assign rows.
  AssignByRow(splitContext, assignContext);
  // 3. Assign blocks.
  AssignByBlock(splitContext, assignContext);
  // 4. Force assignment.
  if (assignContext.coreCache.block == 0 && supportFd_) {
    ForceAssign(splitContext, assignContext);
  }
  result.bN2End[assignContext.curCoreIdx] = assignContext.curBN2Idx;
  result.gS1End[assignContext.curCoreIdx] = assignContext.curS1GIdx;
  result.s2End[assignContext.curCoreIdx] = assignContext.curS2Idx;
  result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);
  assignContext.unassignedCost -= assignContext.coreCache.cost;
  result.maxS2LoopNum =
      std::max(assignContext.coreCache.s2Loop, result.maxS2LoopNum);
  // Record and clear the preceding reduction metadata.
  if (IsNeedRecordFDInfo(assignContext, result)) {
    if (isBatchConsistency_ && remainedBlockNum_ > 0U) {
      // curKvSplitPart already reserves one slot for the core that finishes
      // this row.
      assignContext.curKvSplitPart += remainedBlockNum_ - 1U;
    }
    RecordFDInfo(splitContext, assignContext, result);
    assignContext.preFdDataNum += assignContext.curKvSplitPart;
    assignContext.curKvSplitPart = 1U;
    remainedBlockNum_ = 0U;
  }
  // Update S2 split metadata.
  if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
      assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
    if (isBatchConsistency_) {
      if (IsFirstReductionBlock(assignContext, result)) {
        assignContext.curKvSplitPart++;
      } else {
        assignContext.curKvSplitPart +=
            result.s2End[assignContext.curCoreIdx] -
            result.s2End[assignContext.curCoreIdx - 1U];
      }
      remainedBlockNum_ = assignContext.s1GCache.s1GBlock;
    } else {
      assignContext.curKvSplitPart++;
    }
  }
}

void SmlaVaMetadataCpuKernel::CalcSplitPlan(int64_t costLimit,
                                            const SplitContext& splitContext,
                                            SplitResult& result) {
  const CostInfo& costInfo = splitContext.costInfo;
  const SplitInfo& splitInfo = splitContext.splitInfo;
  if (aicCoreNum_ == 0U) {
    return;
  }
  result.maxCost = 0U;
  result.usedCoreNum = 0U;

  AssignContext assignContext{};
  assignContext.curBIdx = 0U;
  assignContext.curS1GIdx = 0U;
  assignContext.unassignedCost = costInfo.totalCost;
  assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
  assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
  assignContext.bN2S2Loop =
      costInfo.bN2S2LoopOfEachBatch[assignContext.curBIdx];
  CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
  CalcS1GCache(assignContext.curS1GIdx,
               splitContext,
               assignContext.batchCache,
               assignContext.s1GCache);
  assignContext.curS2Idx = assignContext.s1GCache.s2Start;
  // Assign workload.
  for (uint32_t i = 0; i < aicCoreNum_; ++i) {
    if (result.maxCost > costLimit) {
      return;
    }
    if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
      break;
    }
    assignContext.curCoreIdx = i;
    AssignBlocksToCore(splitContext, assignContext, result);
  }
  result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void SmlaVaMetadataCpuKernel::SplitFD(SplitResult& splitRes) {
  // Calculate the total FD data volume.
  uint64_t totalFDLoad = 0;
  for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
    totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
  }
  // Calculate the current number of spare vector cores.
  uint32_t emptyVectorNum = aivCoreNum_ - splitRes.numOfFdHead;
  // Calculate the load handled by each core.
  uint64_t averageLoad = (totalFDLoad + aivCoreNum_ - 1U) /
                         aivCoreNum_;  // Round up to avoid zero load per core.
  uint32_t curCoreIndex = 0;
  for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
    // With no spare vector cores, reduction tasks cannot be split further; use one vector core per reduction task.
    if (emptyVectorNum == 0U) {
      splitRes.fdRes.fdIdx[curCoreIndex] = i;
      splitRes.fdRes.fdMStart[curCoreIndex] = 0U;
      splitRes.fdRes.fdMNum[curCoreIndex] = splitRes.fdRes.fdMSize[i];
      curCoreIndex++;
      continue;
    }
    // Calculate the core count for the current reduction task, rounding down to avoid exceeding the total core count.
    uint32_t curFDVectorNum = splitRes.fdRes.fdS2SplitNum[i] *
                              splitRes.fdRes.fdMSize[i] / averageLoad;
    curFDVectorNum = std::max(1U, curFDVectorNum);
    // Calculate rows per core for the current reduction task, rounding up to avoid zero rows.
    uint32_t curAveMSize =
        (splitRes.fdRes.fdMSize[i] + curFDVectorNum - 1U) / curFDVectorNum;
    curFDVectorNum =
        (splitRes.fdRes.fdMSize[i] + curAveMSize - 1U) / curAveMSize;
    // Use the minimum of the required vector-core count and the currently available count.
    curFDVectorNum =
        std::min(curFDVectorNum, emptyVectorNum + 1U);  // 1: The FD task already owns one core.
    // Assign the FD workload.
    for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
      splitRes.fdRes.fdIdx[curCoreIndex] = i;
      splitRes.fdRes.fdMStart[curCoreIndex] = vid * curAveMSize;
      splitRes.fdRes.fdMNum[curCoreIndex] =
          (vid < curFDVectorNum - 1)
              ? curAveMSize
              : (splitRes.fdRes.fdMSize[i] - vid * curAveMSize);
      curCoreIndex++;
    }
    // Update the spare vector-core count.
    emptyVectorNum -= (curFDVectorNum - 1U);  // 1: The spare count excludes the core already owned by the FD task.
  }
  splitRes.fdRes.fdUsedVecNum = curCoreIndex;
}

bool SmlaVaMetadataCpuKernel::BalanceSchedule(SplitResult& splitRes) {
  SplitContext splitContext(batchSize_);

  // 1. Partition basic blocks and collect metadata.
  CalcSplitInfo(splitContext);
  // Handle a completely empty case.
  if (splitContext.splitInfo.isKvSeqAllZero) {
    splitRes.usedCoreNum = 1U;
    splitRes.bN2End[0] = batchSize_ * numHeadsKv_;
    splitRes.gS1End[0] = 0U;
    splitRes.s2End[0] = 0U;
    return true;
  }
  CalcCostInfo(splitContext);

  // 2. Obtain the assignment plan for each core.
  splitRes.maxCost = INT64_MAX;
  splitRes.usedCoreNum = 1U;

  CalcSplitPlan(splitRes.maxCost, splitContext, splitRes);
  // 3. If FD tasks exist, balance their workload.
  if (splitRes.numOfFdHead > 0U) {
    SplitFD(splitRes);
  }
  splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U);  // Use at least one core.
  return true;
}

bool SmlaVaMetadataCpuKernel::GenMetadata(SplitResult& splitRes) {
  optiling::detail::SmlaMetadata* metadataPtr =
      static_cast<optiling::detail::SmlaMetadata*>(metadata_->GetData());
  *metadataPtr = {};
  // FA Metadata Generate

  for (size_t i = 0; i < aicCoreNum_; ++i) {
    if (i >= splitRes.usedCoreNum) {
      metadataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 0;  // AIC disenable
      continue;
    }
    metadataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 1;  // AIC enable
    // FA START
    metadataPtr->faMetadata[i][FA_BN2_START_INDEX] =
        i == 0 ? 0 : splitRes.bN2End[i - 1];
    metadataPtr->faMetadata[i][FA_M_START_INDEX] =
        i == 0 ? 0 : splitRes.gS1End[i - 1];
    metadataPtr->faMetadata[i][FA_S2_START_INDEX] =
        i == 0 ? 0 : splitRes.s2End[i - 1];
    // FA END
    metadataPtr->faMetadata[i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
    metadataPtr->faMetadata[i][FA_M_END_INDEX] = splitRes.gS1End[i];
    metadataPtr->faMetadata[i][FA_S2_END_INDEX] = splitRes.s2End[i];
    // firstFdDataWorkspace
    metadataPtr->faMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] =
        splitRes.firstFdDataWorkspaceIdx[i];
  }

  // FD Metadata Generate
  for (size_t i = 0; i < aivCoreNum_; ++i) {
    if (i >= splitRes.fdRes.fdUsedVecNum) {
      metadataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 0;  // AIV disenable
      continue;
    }
    metadataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 1;  // AIV enable
    uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
    metadataPtr->fdMetadata[i][FD_BN2_IDX_INDEX] =
        splitRes.fdRes.fdBN2Idx[curFdIdx];
    metadataPtr->fdMetadata[i][FD_M_IDX_INDEX] =
        splitRes.fdRes.fdMIdx[curFdIdx];
    metadataPtr->fdMetadata[i][FD_WORKSPACE_IDX_INDEX] =
        splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
    metadataPtr->fdMetadata[i][FD_WORKSPACE_NUM_INDEX] =
        splitRes.fdRes.fdS2SplitNum[curFdIdx];
    metadataPtr->fdMetadata[i][FD_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
    metadataPtr->fdMetadata[i][FD_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
  }
  return true;
}

namespace {
static const char* kernelType = "SmlaVaMetadata";
REGISTER_CPU_KERNEL(kernelType, SmlaVaMetadataCpuKernel);
}  // namespace

};  // namespace aicpu
