# DESIGN — grouped fat-expert MoE + 3-stage cp.async

Status: phase 1 (cp.async) implemented opt-in; phase 2 (grouped) designed.
Reference: Reederey87 kit (Apache-2.0), `exl3_fat_gemm.cu` / `exl3_fat_moe.cu`.
Scope: `vllm-exl3`, TP2, GLM-5.3-Flash EXL3 4bpw, prefill only.

## Why

At `MAX_NUM_BATCHED_TOKENS=7168`, the E2 fat path in
`overlay/exl3.py::apply_exl3_batched_fat` loops over every expert with
`n_rows > cap` and issues ~15 GPU operations per expert. At 42 layers × ~10.5
fat experts, that is ~441 expert invocations and ~6,600 dispatches per engine
step, plus a Python loop.

The dominant cost is not launch overhead (~26–132 ms/step). It is the
intermediate global-memory traffic: `h`, `h13`, `gate_up`, `act`, `act_h`,
`h2`, and the two trellis copies per expert. The reference measured ~61 % fewer
bytes per row with a fused gather → gate/up → down chain, an isolated 1.87×
path speedup, and **+19.6 % cold prefill at 240k**.

## Phase 1 — 3-stage cp.async (implemented, opt-in)

`overlay/exl3_fat_gemm_pipe.cu` is a copy of the reference S2b kernel with
local `cp.async` wrappers and unique public symbols:

- `FAT_STAGES = 3`, per-stage `sh_a` + `sh_b`, issue-before-wait;
- `cp.async.cg.shared.global` 16 B loads, `commit_group`, `wait_group`;
- OOB A rows are zero-filled **synchronously** (`*a_dst = int4{}`); do not use
  `cp_async_pred` (Blackwell miscompile);
- same ABI and semantics as `exl3_fat_gemm`; selected by
  `EXL3_FAT_PIPELINE=1`, default 0;
- build: `patch_exl3_fat_kernel.py` copies both pairs and binds
  `exl3_fat_gemm_pipe` / `_scatter`; Dockerfile asserts the symbols.

Reference receipt: **+38.6 / +41.4 / +40.8 %** kernel throughput at M=3584,
bit-exact over 56 shapes, 95 regs, 0 spills, 2 CTAs/SM. E2E gain is smaller
because the fat GEMM is a share of prefill; measure the cold ladder.

### Gates

- bitwise `torch.equal` vs stock over M ∈ {1,17,63,64,65,127,128,129,1024,
  3584}, K ∈ {16,128,1024,4096}, N ∈ {128,1024,4096};
- `compute-sanitizer` memcheck/initcheck/synccheck: 0 errors;
- microbench M=3584 ≥ 1.25× (target +38–41 %);
- cold 8k/100k/300k non-inferior; decode ±2 %; pool/hits unchanged.

## Phase 2 — grouped dispatch (designed, not wired)

New `overlay/exl3_fat_moe.cu/.cuh` with three launchers:

- `exl3_fat_moe_gather` — `h13[row] = had128(x[token[row]] * gate.suh[expert])`;
- `exl3_fat_moe_gateup` — two B streams (gate/up), fused SwiGLU + fp16
  rounding + `down.suh` pre-scale + Hadamard → `h2`;
- `exl3_fat_moe_down` — two adjacent 128-column halves, fused Hadamard +
  `svh` + route weight + `float4 atomicAdd` scatter.

The work queue is built on device from `counts` (cumsum/searchsorted), with
64-row segments and a capacity-sized grid; no `counts.tolist()` and no host
sync. Python branch in `apply_exl3_fused_moe` before `_stage_counts_to_host`:

```python
if grouped_fat_enabled() and not use_row_tiles and effective_tier == "grouped":
    _exl3_moe_launch(...)          # thin experts only
    apply_exl3_grouped_fat(...)    # 3 grouped kernels
    return out
```

New env `EXL3_FAT_GROUPED=0|1` (default 0). Ineligible checkpoint → E2
`kernel` with a recorded reason; missing symbols with `=1` → fail closed at
load. Row tiles preempt grouped.

### Gates

- gather vs E2 chain **bitwise**; gateup single-expert **bitwise**; down
  single-expert **bitwise**;
- multi-expert down is not bitwise (atomic order); gate maxabs ≤ 0.125,
  nrmse ≤ 5e-7 over 20 repeats;
- TP2 sum parity; zero-fill/tail rows; no IMA; decode bitwise unchanged;
- kernel microbench ≥1.5× at M=512/1024/2048, parity at M=128;
- cold 300k ≥ +5 %, 100k ≥ +3 %, 8k ≥ −2 %, decode ±2 %, pool/hits unchanged;
- `grouped_calls == 3 × 42 × chunks`; `grouped_scratch_growths <= 1`;
- scratch 560 MiB/rank at 57,344 rows; monitor MemFree/pool.

## Risks

- exllamav3 ABI: user pin `c5d9c657` vs reference v1.4.9; local cp.async
  wrappers and a bitwise gate first.
- SMEM/occupancy: 32 KB dynamic, `REG:128` → 2 CTAs/SM; do not reopen the
  occupancy sweep without a new profile.
- Blackwell `cp.async` zero-fill: use the 4-operand `src-size=0` form or a
  synchronous zero-fill; never `cp_async_pred`.
- Build time: one integrated rebuild; keep Python overlay edits in a later
  layer.
- License: Apache-2.0 headers and `NOTICE`; no post-`cf0f430` AGPL code.

## Order

1. Freeze baseline receipts (cold 8k/100k/300k, decode, pool, microbench).
2. Ship phase 1 (`EXL3_FAT_PIPELINE=1` A/B), keep default 0 until green.
3. Add grouped kernels + build wiring, default 0.
4. Python integration + CPU tests, default 0.
5. Enable, qualify, then flip the default only with a receipt.
