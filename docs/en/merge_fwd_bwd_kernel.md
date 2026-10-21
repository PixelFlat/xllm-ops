<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MergeFwdBwdKernel Operator Design

English | [简体中文](../zh/merge_fwd_bwd_kernel.md)

## Product support

| Product | Supported |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | Yes |
| Atlas A3 training/inference products | Yes |
| Atlas A2 training/inference products | Yes |

## Overview

`MergeFwdBwdKernel` merges the per-rank affine-chain summaries produced by
[PreProcessFwdKernelMerged](pre_process_fwd_kernel_merged.md) into a single
head state, implementing the composition step of KDA/GDN context-parallel
prefill:

```text
h ← He_0
h ← M_i @ h + He_i    for i = 1 .. N-1
```

`He` is the leading `V` columns of the gathered `ag_hm` and `M` is the
trailing `K` columns. Both stay `[K, V]` and `[K, K]`; the GEMMs run in that
order. `state_v_first` only changes the physical column order of the
caller-owned `h` buffer.

Like the FLA reference `merge_fwd_bwd_kernel`, there is a single `h` buffer:
the CP path does not read the original value, the result is written back in
place, and the operator IR registers the same `h` as both input and output so
graph compilation observes the write.

## Symbols and shapes

```text
S  = number of participating ranks (ag_hm batch dim)
HV = value heads; K = V = 128 (fixed)
```

| Name | Direction | Data type | Shape | Description |
| --- | --- | --- | --- | --- |
| `h` | Input/Output | FP32/BF16 | `[HV,128,128]` | In-place output buffer; must be contiguous |
| `ag_hm` | Input | FP32/BF16 | `[S,HV,128,256]` | Per-rank `He` (first 128 columns) and `M` (last 128 columns); may be non-contiguous (host gathers first) |

Attributes:

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `forward` | BOOL | true | true: merge from `rank - N` towards `rank`; false: reverse |
| `rank` | INT64 | 0 | Current rank, in `[0, S)` |
| `preOrPostNumRanks` | INT64 | 0 | Number of ranks `N` participating in the merge; `N >= 1` |
| `state_v_first` | BOOL | false | false: `h` tail dims are `[K, V]`; true: `[V, K]` |

## aclnn interface

```cpp
aclnnStatus aclnnMergeFwdBwdKernelGetWorkspaceSize(
    const aclTensor *h, const aclTensor *agHm,
    int64_t preOrPostNumRanks, int64_t rank, bool forward, bool stateVFirst,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnMergeFwdBwdKernel(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);
```

Full semantics and constraints are documented in the operator tree:
[`merge_fwd_bwd_kernel/docs/api.md`](../../xllm_ops/merge_fwd_bwd_kernel/docs/api.md)
(design notes:
[`docs/design.md`](../../xllm_ops/merge_fwd_bwd_kernel/docs/design.md)).

## Implementation notes

- AscendC kernel with per-arch sources (`arch22` for Atlas A2/A3, `arch35`
  for Ascend 950); registered via the `aclnn_exclude` set and exported from
  the vendor `libcust_opapi.so`.
- The composition runs in the `ag_hm` dtype (FP32 in the xLLM KDA CP path,
  matching the summary operator's FP32 chain precision).
