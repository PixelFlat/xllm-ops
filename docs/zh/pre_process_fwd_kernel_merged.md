<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# PreProcessFwdKernelMerged 算子设计

[English](../en/pre_process_fwd_kernel_merged.md) | 简体中文

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

`PreProcessFwdKernelMerged` 是 KDA/GDN context parallel（CP）prefill 方案的
summary-pass 算子。它把一个打包的 CP 窗口压缩为每序列、每 value head 一条
**仿射链**表示 `hm = [h | m]`：`h`（`K×V`）是该段对递推状态的贡献，
`m`（`K×K`）是作用于传入先验状态的线性映射。链可结合复合：

```text
(M2, E2) ∘ (M1, E1) = (M2 @ M1, M2 @ E1 + E2)
```

因此每个 CP rank 可以独立总结本地窗口，由编排层收集后用
[MergeFwdBwdKernel](merge_fwd_bwd_kernel.md) 合并。跨 rank 的 `all_gather`
与合并不在本算子内。

支持的算法族：GDN（标量 gate `g`）与 KDA（逐通道 gate `gk`）；DPLR 不支持
（`v`/`bg` 必须为空）。

本算子是 xLLM KDA CP prefill 路径三个算子之一，另见
[XllmChunkKdaFwdPrepare](xllm_chunk_kda_fwd_prepare.md)。

## 符号与形状

```text
Nseq = 窗口内段数（len(cu_seqlens) - 1）
T    = 窗口 token 数；K = V = 128；chunk_size = 64（固定）
HK   = key head 数；HV = value head 数（GVA：HV ≥ HK 且 HV % HK == 0）
```

| 名称 | 方向 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `k` | 输入 | BF16 | `[1,HK,T,K]` | raw key |
| `w` | 输入 | BF16 | `[1,HV,T,K]` | erase/WY 因子（`h` 与 `m` 的左矩阵） |
| `u` | 输入 | BF16 | `[1,HV,T,V]` | 取值来源 |
| `g` | 输入 | FP32/BF16 | `[1,HV,T]` | 标量对数衰减（base-2、chunk 内累积）；与 `gk` 二选一 |
| `gk` | 输入 | FP32/BF16 | `[1,HV,T,K]` | 逐通道对数衰减（base-2、chunk 内累积）；与 `g` 二选一 |
| `cu_seqlens` | 输入 | INT64（host） | `[Nseq+1]` | 严格递增的段边界；允许打包窗口的子区间 |
| `hm` | 输出 | FP32 | `[Nseq,HV,K,V+K]` | 每链仿射摘要：左 `V` 列为 `h`，右 `K` 列为 `m` |

属性：`chunk_size`（INT64，本版固定 64，其它值 host 拦截）。

## aclnn 接口

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

`cu_seqlens` 是 host 侧整型数组（不是 device tensor）：边界校验与每段
`bos/T_win/NT` 的展开都在 host tiling 内完成。完整语义、逐 Stage 设计与已知
限制见算子目录内文档：
[`pre_process_fwd_kernel_merged/docs/api.md`](../../xllm_ops/pre_process_fwd_kernel_merged/docs/api.md)
（设计说明：
[`docs/design.md`](../../xllm_ops/pre_process_fwd_kernel_merged/docs/design.md)）。

## 实现说明

- 仿射链严格按 FP32 累加（`ieee` 精度模式，不用 tf32x3），这是跨 rank
  合并可逐位复现的基础。
- host tiling 实现 colSplit/hybridS 列切分与混合调度启发式（由 AIC 核数
  推导）；调度只改变任务划分，不改变 FP32 累加顺序。
- AscendC kernel，按架构分目录（`arch22` 对应 Atlas A2/A3，`arch35` 对应
  Ascend 950）；通过 `aclnn_exclude` 集合注册，符号从 vendor
  `libcust_opapi.so` 导出。
