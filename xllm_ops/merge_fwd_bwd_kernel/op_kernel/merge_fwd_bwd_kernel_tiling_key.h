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

#ifndef MERGE_FWD_BWD_KERNEL_TILING_KEY_H
#define MERGE_FWD_BWD_KERNEL_TILING_KEY_H

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace MergeFwBwd {

#define MERGE_FWD_BWD_KERNEL_TPL_BF16 10
#define MERGE_FWD_BWD_KERNEL_TPL_FP32 30

#ifndef TORCH_MODE
ASCENDC_TPL_ARGS_DECL(MergeFwdBwdKernel,
                      ASCENDC_TPL_DTYPE_DECL(D_T,
                                             MERGE_FWD_BWD_KERNEL_TPL_BF16,
                                             MERGE_FWD_BWD_KERNEL_TPL_FP32), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T,
                                               MERGE_FWD_BWD_KERNEL_TPL_BF16)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T, MERGE_FWD_BWD_KERNEL_TPL_FP32)), );
#endif

}  // namespace MergeFwBwd

#endif
