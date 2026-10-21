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
 * \file pre_process_fwd_kernel_merged_kernel.h
 * \brief arch 分派：__CCE_AICORE__ == 310（A5/950）→ arch35/；否则（A2/A3）→
 * arch22/。
 */

#ifndef PREF_PROCESS_FWD_KERNEL_MERGED_KERNEL_H
#define PREF_PROCESS_FWD_KERNEL_MERGED_KERNEL_H

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/pre_process_fwd_kernel_merged_cube.h"
#include "arch35/pre_process_fwd_kernel_merged_vector.h"
#else
#include "arch22/pre_process_fwd_kernel_merged_cube.h"
#include "arch22/pre_process_fwd_kernel_merged_vector.h"
#endif

#endif  // PREF_PROCESS_FWD_KERNEL_MERGED_KERNEL_H
