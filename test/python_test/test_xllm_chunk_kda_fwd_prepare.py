# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Smoke test for the XllmChunkKdaFwdPrepare vendor symbols.

The KDA CP operators are registered through the ``aclnn_exclude`` set: their
entry symbols are exported from the installed vendor ``libcust_opapi.so``
and are intentionally not wired into the CANN ``torch.ops`` registry (the
engine-side loader owns the calling convention). This smoke test therefore
verifies, on an NPU environment, that a standard-path installed vendor
exports ``aclnnXllmChunkKdaFwdPrepare`` together with its workspace-size
query and the base aclnn plumbing.
"""

import ctypes
import os

import pytest

torch_npu = pytest.importorskip("torch_npu")


_SYMBOLS = (
    "aclnnXllmChunkKdaFwdPrepare",
    "aclnnXllmChunkKdaFwdPrepareGetWorkspaceSize",
    "aclCreateTensor",
    "aclDestroyTensor",
)


def _has_npu():
    import torch

    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")


def _candidate_libraries():
    """Standard opp vendor roots, in CANN discovery order."""
    roots = []
    home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend")
    vendors = os.path.join(home, "opp", "vendors")
    if os.path.isdir(vendors):
        for name in sorted(os.listdir(vendors)):
            lib = os.path.join(vendors, name, "op_api", "lib", "libcust_opapi.so")
            if os.path.isfile(lib):
                roots.append(lib)
    for extra in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if extra:
            lib = os.path.join(extra, "op_api", "lib", "libcust_opapi.so")
            if os.path.isfile(lib):
                roots.append(lib)
    return roots


def test_prepare_symbols_exported():
    found = []
    for lib_path in _candidate_libraries():
        try:
            mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
            lib = ctypes.CDLL(lib_path, mode=mode)
        except OSError:
            continue
        try:
            for symbol in _SYMBOLS:
                getattr(lib, symbol)
            found.append(lib_path)
        except AttributeError:
            continue
    assert found, (
        "no installed vendor exports aclnnXllmChunkKdaFwdPrepare; expected "
        "the custom_xllm_math standard-path install (bash build.sh "
        "-n xllm_chunk_kda_fwd_prepare)"
    )
