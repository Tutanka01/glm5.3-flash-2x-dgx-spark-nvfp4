#pragma once

#include <torch/extension.h>

// Opt-in 3-stage cp.async variant of exl3_fat_gemm. Same ABI and semantics as
// the stock per-expert kernel; selected by EXL3_FAT_PIPELINE=1.
void exl3_fat_gemm_pipe(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_fat_gemm_pipe_scatter(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    at::Tensor token_idx,
    at::Tensor route_weight,
    int64_t K,
    bool mcg,
    bool mul1);
