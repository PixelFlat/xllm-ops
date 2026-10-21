<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# XllmChunkKdaFwdPrepare 算子设计

[English](../en/xllm_chunk_kda_fwd_prepare.md) | 简体中文

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

`XllmChunkKdaFwdPrepare` 是 KDA（Kimi Delta Attention）prefill 路径的
chunk 预处理算子。输入投影后的 `q/k/v` 流、逐通道衰减 gate `g` 与更新
gate `beta`，单个融合 kernel 产出后续 chunk 递推所需的全部 chunk 级中间量：
FP32 衰减后的 key gate `gk`、内积 `Aqk/Akk`、WY 表示因子 `w/u`、加 gate 的
`qg/kg/qg_scaled`、L2 归一化后的 `q_hat/k_hat` 及其倒数标准差
`q_rstd/k_rstd`、有效更新 gate `beta_eff`。

算子支持 `BNSD/BSND/NTD/TND` 四种布局，通过 `cu_seqlens/chunk_indices`
支持变长打包流，并把 kernel 内 QK L2 归一化、gate/beta 激活、exp2 衰减
模式、safe gate 钳制等编译期开关以运行时属性暴露。`K = V = 128`、
`chunk_size = 64` 为固定规格。

本算子是 xLLM KDA CP（context parallel）prefill 路径三个算子之一，另见
[PreProcessFwdKernelMerged](pre_process_fwd_kernel_merged.md) 与
[MergeFwdBwdKernel](merge_fwd_bwd_kernel.md)。

## 符号与形状

```text
B  = 序列数（变长模式下由 cu_seqlens 描述）
T  = 打包 token 总数
HK = key 侧 head 数
HV = value 侧 head 数（GVA：0 < HK <= HV 且 HV % HK == 0）
K  = 128（key head 维），V = 128（value head 维）
```

主要输入（打包布局，物理顺序由 `layout` 属性选择）：

| 名称 | 方向 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `q` | 输入 | BF16 | `[B,HK,T,K]` | Query 流 |
| `k` | 输入 | BF16 | `[B,HK,T,K]` | Key 流 |
| `v` | 输入 | BF16 | `[B,HV,T,V]` | Value 流 |
| `g` | 输入 | BF16/FP32 | `[B,HV,T,K]` | 逐通道对数衰减 gate |
| `beta` | 输入 | BF16/FP32 | `[B,HV,T]` | 更新 gate |
| `a_log` | 输入 | FP32 | `[HV]` | 可选；kernel 内 gate 模式必传 |
| `dt_bias` | 输入 | FP32 | `[HV*128]` | 可选 softplus 偏置 |
| `cu_seqlens` | 输入 | INT64（host） | `[B+1]` | 可选变长边界 |
| `chunk_indices` | 输入 | INT64（host） | 打包 | 可选 sequence-major chunk 索引 |

13 个输出在算子 IR 层全部为 `REQUIRED`；aclnn launcher 允许调用方对不需要
的输出传空 descriptor（稳定 ABI 层的四种反向模式 `none/forward/recompute/save`）。
固定 dtype：`gk/q_rstd/k_rstd/beta_eff` 为 FP32，
`Aqk/Akk/w/u/qg/kg/qg_scaled/q_hat/k_hat` 为 BF16。

## aclnn 接口

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

详细的参数约束、反向模式输出槽位表与返回码见算子目录内文档：
[`xllm_chunk_kda_fwd_prepare/docs/api.md`](../../xllm_ops/xllm_chunk_kda_fwd_prepare/docs/api.md)
（设计说明：
[`docs/design.md`](../../xllm_ops/xllm_chunk_kda_fwd_prepare/docs/design.md)）。

## 实现说明

- AscendC kernel，按架构分目录：`op_kernel/arch22`（Atlas A2/A3）与
  `op_kernel/arch35`（Ascend 950），由 `__CCE_AICORE__` 选择。
- host tiling 同时支持固定 batch 与变长 chunk 流；变长路径要求段长为
  64 的整数倍。
- 通过 `aclnn_exclude` 集合注册：符号从 vendor `libcust_opapi.so` 导出，
  不接入 CANN `torch.ops` 注册表，由 xLLM 引擎侧的 KDA CP 加载器调用。
