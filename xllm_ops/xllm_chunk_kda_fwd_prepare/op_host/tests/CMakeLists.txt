# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").
# -----------------------------------------------------------------------------------------------------------

add_executable(
    xllm_chunk_kda_fwd_prepare_tiling_processor_test
    xllm_chunk_kda_fwd_prepare_tiling_processor_test.cpp
)
target_compile_features(
    xllm_chunk_kda_fwd_prepare_tiling_processor_test
    PRIVATE cxx_std_17
)

add_test(
    NAME xllm_chunk_kda_fwd_prepare_tiling_processor_test
    COMMAND xllm_chunk_kda_fwd_prepare_tiling_processor_test
)
