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

#ifndef ACLNN_PLATFORM_H_
#define ACLNN_PLATFORM_H_

#include <set>

#include "opdev/platform.h"

namespace Ops {
namespace Transformer {
namespace AclnnUtil {

using namespace op;

/**
 * 检查当前芯片架构是否为RegBase
 */
inline static bool IsRegbase(NpuArch npuArch) {
  const static std::set<NpuArch> regbaseNpuArchs = {NpuArch::DAV_3510,
                                                    NpuArch::DAV_5102};
  return regbaseNpuArchs.find(npuArch) != regbaseNpuArchs.end();
}

inline static bool IsRegbase() {
  auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
  return IsRegbase(npuArch);
}

}  // namespace AclnnUtil
}  // namespace Transformer
}  // namespace Ops

#endif
