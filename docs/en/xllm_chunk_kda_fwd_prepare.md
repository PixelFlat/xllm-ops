<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# XllmChunkKdaFwdPrepare Operator Design

English | [简体中文](../zh/xllm_chunk_kda_fwd_prepare.md)

## Product support

| Product | Supported |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | Yes |
| Atlas A3 training/inference products | Yes |
| Atlas A2 training/inference products | Yes |

## Overview

`XllmChunkKdaFwdPrepare` is the chunked pre-processing operator of the KDA
(Kimi Delta Attention) prefill path. Given the projected `q/k/v` streams, the
per-channel decay gate `g` and the update gate `beta`, it produces all
chunk-level intermediates that the downstream chunk recurrence consumes in one
fused kernel: the FP32 decayed key gate `gk`, the inner products `Aqk/Akk`,
the WY-representation factors `w/u`, the gated queries/keys `qg/kg/qg_scaled`,
the L2-normalized `q_hat/k_hat` with their reciprocal standard deviations
`q_rstd/k_rstd`, and the effective update gate `beta_eff`.

The operator is layout-flexible (`BNSD/BSND/NTD/TND`), supports varlen packed
streams via `cu_seqlens`/`chunk_indices`, and exposes compile-time switches
(QK L2-norm in kernel, gate/beta activation in kernel, exp2 decay mode, safe
gate clamping) as runtime attributes. `K = V = 128` and `chunk_size = 64` are
fixed.

It is one of the three operators that back the xLLM KDA context-parallel (CP)
prefill path; see also [PreProcessFwdKernelMerged](pre_process_fwd_kernel_merged.md)
and [MergeFwdBwdKernel](merge_fwd_bwd_kernel.md).

## Symbols and shapes

```text
B  = number of sequences (varlen mode: described by cu_seqlens)
T  = total number of packed tokens
HK = number of key heads
HV = number of value heads (GVA: 0 < HK <= HV, HV % HK == 0)
K  = 128 (key head dim), V = 128 (value head dim)
```

Main inputs (packed layouts, `layout` attribute selects the physical order):

| Name | Direction | Data type | Shape | Description |
| --- | --- | --- | --- | --- |
| `q` | Input | BF16 | `[B,HK,T,K]` | Query stream |
| `k` | Input | BF16 | `[B,HK,T,K]` | Key stream |
| `v` | Input | BF16 | `[B,HV,T,V]` | Value stream |
| `g` | Input | BF16/FP32 | `[B,HV,T,K]` | Per-channel log-decay gate |
| `beta` | Input | BF16/FP32 | `[B,HV,T]` | Update gate |
| `a_log` | Input | FP32 | `[HV]` | Optional; required by in-kernel gate mode |
| `dt_bias` | Input | FP32 | `[HV*128]` | Optional softplus bias |
| `cu_seqlens` | Input | INT64 (host) | `[B+1]` | Optional varlen boundaries |
| `chunk_indices` | Input | INT64 (host) | packed | Optional sequence-major chunk index |

The 13 outputs are all `REQUIRED` at the operator-IR level; the aclnn launcher
accepts null descriptors for outputs the caller does not need (four backward
modes: `none/forward/recompute/save` in the stable ABI layer). Fixed dtypes:
FP32 for `gk/q_rstd/k_rstd/beta_eff`, BF16 for `Aqk/Akk/w/u/qg/kg/qg_scaled/q_hat/k_hat`.

## aclnn interface

```cpp
aclnnStatus aclnnXllmChunkKdaFwdPrepareGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *g, const aclTensor *beta,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    const char *layout, double scale, int64_t chunkSize, double epsilon,
    bool useQkL2normInKernel, bool useGateInKernel, bool useBetaSigmoidInKernel,
    bool allowNegEigval, bool safeGate, double lowerBound, bool useExp2,
    const aclTensor *gkOut, const aclTensor *aqkOut, const aclTensor *akkOut,
    const aclTensor *wOut, const aclTensor *uOut, const aclTensor *qgOut,
    const aclTensor *kgOut, const aclTensor *qgScaledOut, const aclTensor *qHatOut,
    const aclTensor *kHatOut, const aclTensor *qRstdOut, const aclTensor *kRstdOut,
    const aclTensor *betaEffOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnXllmChunkKdaFwdPrepare(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);
```

Detailed parameter constraints, the backward-mode output-slot table and
return codes are documented in the operator tree:
[`xllm_chunk_kda_fwd_prepare/docs/api.md`](../../xllm_ops/xllm_chunk_kda_fwd_prepare/docs/api.md)
(design notes:
[`docs/design.md`](../../xllm_ops/xllm_chunk_kda_fwd_prepare/docs/design.md)).

## Implementation notes

- AscendC kernel with per-arch sources: `op_kernel/arch22` (Atlas A2/A3) and
  `op_kernel/arch35` (Ascend 950), selected by `__CCE_AICORE__`.
- Host-side tiling supports both fixed batch and varlen chunked streams; the
  varlen path requires spans to be multiples of 64 tokens.
- Registered via the `aclnn_exclude` generation set: the symbols are exported
  from the vendor `libcust_opapi.so` and are not wired into the CANN
  `torch.ops` registry. The xLLM engine loads them through its own KDA CP op
  loader.
