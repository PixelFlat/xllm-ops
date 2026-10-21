/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*!
 * \file pre_process_fwd_kernel_merged_tiling.h
 * \brief Tiling data 复用 kernel 侧的纯结构体定义（保证 host / device
 * 字段一致）。
 */

#pragma once

#include "../op_kernel/pre_process_fwd_kernel_merged_struct.h"

namespace optiling {

using GDN::PreProcessFwdKernelMergedTilingData;

struct PreProcessFwdKernelMergedCompileInfo {};

}  // namespace optiling
