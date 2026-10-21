<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MergeFwdBwdKernel 算子设计

[English](../en/merge_fwd_bwd_kernel.md) | 简体中文

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

`MergeFwdBwdKernel` 把 [PreProcessFwdKernelMerged](pre_process_fwd_kernel_merged.md)
产出的各 rank 仿射链摘要合并成单个 head 状态，实现 KDA/GDN context
parallel prefill 的复合步骤：

```text
h ← He_0
h ← M_i @ h + He_i    for i = 1 .. N-1
```

`He` 是收集到的 `ag_hm` 的前 `V` 列，`M` 是后 `K` 列；两者保持 `[K, V]` 与
`[K, K]`，GEMM 按此顺序执行。`state_v_first` 只改变调用方 `h` 缓冲的物理
列序。

与 FLA 参考 `merge_fwd_bwd_kernel` 一致，只有一块 `h`：CP 路径不读取原值，
结果原地写回；算子 IR 将同一个 `h` 同时登记为输入与输出，使图编译能看到
这次写。

## 符号与形状

```text
S  = 参与合并的 rank 数（ag_hm 的 batch 维）
HV = value head 数；K = V = 128（固定）
```

| 名称 | 方向 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `h` | 输入/输出 | FP32/BF16 | `[HV,128,128]` | 原地输出缓冲；必须连续 |
| `ag_hm` | 输入 | FP32/BF16 | `[S,HV,128,256]` | 各 rank 的 `He`（前 128 列）与 `M`（后 128 列）；可不连续（host 先收连续） |

属性：

| 名称 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `forward` | BOOL | true | true：从 `rank - N` 向 `rank` 合并；false：反向 |
| `rank` | INT64 | 0 | 当前 rank，范围 `[0, S)` |
| `preOrPostNumRanks` | INT64 | 0 | 参与合并的 rank 数 `N`；`N >= 1` |
| `state_v_first` | BOOL | false | false：`h` 末两维为 `[K, V]`；true：`[V, K]` |

## aclnn 接口

```cpp
aclnnStatus aclnnMergeFwdBwdKernelGetWorkspaceSize(
    const aclTensor *h, const aclTensor *agHm,
    int64_t preOrPostNumRanks, int64_t rank, bool forward, bool stateVFirst,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnMergeFwdBwdKernel(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);
```

完整语义与约束见算子目录内文档：
[`merge_fwd_bwd_kernel/docs/api.md`](../../xllm_ops/merge_fwd_bwd_kernel/docs/api.md)
（设计说明：
[`docs/design.md`](../../xllm_ops/merge_fwd_bwd_kernel/docs/design.md)）。

## 实现说明

- AscendC kernel，按架构分目录（`arch22` 对应 Atlas A2/A3，`arch35` 对应
  Ascend 950）；通过 `aclnn_exclude` 集合注册，符号从 vendor
  `libcust_opapi.so` 导出。
- 复合按 `ag_hm` 的 dtype 执行（xLLM KDA CP 路径为 FP32，与 summary 算子的
  FP32 链精度一致）。
