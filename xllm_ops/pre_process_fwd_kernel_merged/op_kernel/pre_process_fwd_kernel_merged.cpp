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

#include "pre_process_fwd_kernel_merged_kernel.h"
#include "pre_process_fwd_kernel_merged_tiling_key.h"

// ---- 入口常量自检（§4.4 ②/§7.7：常量集中处配 static_assert）----
// tiling 结构由 host 写入、kernel 解析，两侧必须同源同尺寸
static_assert(sizeof(GDN::PreProcessFwdKernelMergedTilingData) <= 4096,
              "tiling 结构超过预留容量，请检查 host/kernel 是否同源");
static_assert(sizeof(GDN::PreProcessFwdKernelMergedTilingData) % 8 == 0,
              "tiling 结构需按 8B 对齐（host 侧写入按 int64 序列化）");
// 每个工作项一块 workspace：按 512B 对齐，供 UB<->GM 的 DataCopy 落点使用
static_assert(GDN::PPFM_CORE_WS_BYTES % 512 == 0,
              "每核 workspace 尺寸需按 512B 对齐");
#ifndef TORCH_MODE
// GATE_MODE 由 *_tiling_key.h 的 ASCENDC_TPL_SEL 实例化（1/2/3）；与 host 侧
// SetTilingKey(gateMode+1) 一一对应。kernel 内部仍读 tiling->gateMode，两者由
// gate 保证一致。
template <int GATE_MODE>
__global__ __aicore__ void pre_process_fwd_kernel_merged(GM_ADDR k,
                                                         GM_ADDR w,
                                                         GM_ADDR u,
                                                         GM_ADDR g,
                                                         GM_ADDR gk,
                                                         GM_ADDR bg,
                                                         GM_ADDR v,
                                                         GM_ADDR cu_seqlens,
                                                         GM_ADDR hm,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling) {
  static_assert(GATE_MODE >= PPFM_TPL_GATE_G && GATE_MODE <= PPFM_TPL_GATE_BG,
                "非法 GATE_MODE TilingKey");
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  REGISTER_TILING_DEFAULT(GDN::PreProcessFwdKernelMergedTilingData);
  GET_TILING_DATA_WITH_STRUCT(
      GDN::PreProcessFwdKernelMergedTilingData, tilingData, tiling);
  GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }
  (void)bg;  // DPLR（USE_BG）不支持：host 侧已拒绝非空 bg，这里只是占位
  GDN::PpFwdCtx ctx;
  ctx.k = k;
  ctx.w = w;
  ctx.u = u;
  ctx.g = g;
  ctx.gk = gk;
  ctx.v = v;
  ctx.cu = cu_seqlens;
  ctx.hm = hm;
  ctx.ws = userWS;
  ctx.tiling = &tilingData;
  if ASCEND_IS_AIC {
    GDN::PpFwdCube cube(ctx);
    cube.Run();
  }
  if ASCEND_IS_AIV {
    GDN::PpFwdVector vec(ctx);
    vec.Run();
  }
}
#endif
