<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# PreProcessFwdKernelMerged Operator Design

English | [简体中文](../zh/pre_process_fwd_kernel_merged.md)

## Product support

| Product | Supported |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | Yes |
| Atlas A3 training/inference products | Yes |
| Atlas A2 training/inference products | Yes |

## Overview

`PreProcessFwdKernelMerged` is the summary-pass operator of the KDA/GDN
context-parallel (CP) prefill scheme. It compresses one packed CP window into
an **affine chain** representation `hm = [h | m]` per sequence and per value
head, where `h` (`K x V`) is the span's contribution to the recurrent state
and `m` (`K x K`) is the linear map applied to the incoming prior state.
Chains compose associatively across ranks:

```text
(M2, E2) ∘ (M1, E1) = (M2 @ M1, M2 @ E1 + E2)
```

so each CP rank can summarize its local window independently and the
orchestration layer merges the per-rank chains with
[MergeFwdBwdKernel](merge_fwd_bwd_kernel.md). The cross-rank `all_gather` and
merge happen outside this operator.

Supported algorithm families: GDN (scalar gate `g`) and KDA (per-channel gate
`gk`). DPLR is not supported (`v`/`bg` must be null).

It is one of the three operators that back the xLLM KDA CP prefill path; see
also [XllmChunkKdaFwdPrepare](xllm_chunk_kda_fwd_prepare.md).

## Symbols and shapes

```text
Nseq = number of segments in the window (len(cu_seqlens) - 1)
T    = window token count; K = V = 128; chunk_size = 64 (fixed)
HK   = key heads; HV = value heads (GVA: HV >= HK, HV % HK == 0)
```

| Name | Direction | Data type | Shape | Description |
| --- | --- | --- | --- | --- |
| `k` | Input | BF16 | `[1,HK,T,K]` | Raw key |
| `w` | Input | BF16 | `[1,HV,T,K]` | Erase/WY factor (left matrix of `h` and `m`) |
| `u` | Input | BF16 | `[1,HV,T,V]` | Value source |
| `g` | Input | FP32/BF16 | `[1,HV,T]` | Scalar log-decay (base-2, chunk-cumulative); mutually exclusive with `gk` |
| `gk` | Input | FP32/BF16 | `[1,HV,T,K]` | Per-channel log-decay (base-2, chunk-cumulative); mutually exclusive with `g` |
| `cu_seqlens` | Input | INT64 (host) | `[Nseq+1]` | Strictly increasing segment boundaries; sub-intervals of the packed window are allowed |
| `hm` | Output | FP32 | `[Nseq,HV,K,V+K]` | Per-chain affine summary: left `V` columns are `h`, right `K` columns are `m` |

Attribute: `chunk_size` (INT64, fixed at 64 in this version; other values are
rejected on the host).

## aclnn interface

```cpp
aclnnStatus aclnnPreProcessFwdKernelMergedGetWorkspaceSize(
    const aclTensor *k, const aclTensor *w, const aclTensor *u,
    const aclTensor *gOptional, const aclTensor *gkOptional,
    const aclTensor *vOptional, const aclTensor *bgOptional,
    const aclIntArray *cuSeqlens, int64_t chunkSize,
    const aclTensor *hmOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnPreProcessFwdKernelMerged(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);
```

`cu_seqlens` is a host-side integer array (not a device tensor): boundary
validation and per-segment `bos/T_win/NT` expansion all happen in host
tiling. The full semantics, per-stage design and known limitations are
documented in the operator tree:
[`pre_process_fwd_kernel_merged/docs/api.md`](../../xllm_ops/pre_process_fwd_kernel_merged/docs/api.md)
(design notes:
[`docs/design.md`](../../xllm_ops/pre_process_fwd_kernel_merged/docs/design.md)).

## Implementation notes

- The affine chain accumulates strictly in FP32 (`ieee` precision mode; tf32x3
  is not used), which is what makes cross-rank merges bit-reproducible.
- Host tiling implements column-split and hybrid scheduling heuristics
  (colSplit/hybridS) derived from the AIC core count; the scheduling changes
  task partitioning only and never changes the FP32 accumulation order.
- AscendC kernel with per-arch sources (`arch22` for Atlas A2/A3, `arch35`
  for Ascend 950); registered via the `aclnn_exclude` set and exported from
  the vendor `libcust_opapi.so`.
