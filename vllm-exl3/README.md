# GLM-5.3-Flash EXL3 — vLLM lane for 2× DGX Sparks

This is the **EXL3 / vLLM lane** of the repo: a vendored, hardened copy of the
[MiaAI-Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
recipe (MIT), merged with this repo's validation culture and its public-issue
fixes. The sibling lane (`../README.md`, SGLang + NVFP4) stays untouched; the
two are alternative products over the same hardware, not competitors to
rewrite.

> **Status: bring-up ready, not yet hardware-validated on this kit.** The
> serve path is upstream's, reproduced upstream on a second GB10 pair. The
> changes this vendor adds are host-side (bring-up robustness + tooling) and
> are covered by `tests/run-local.sh`; the on-cluster protocol to validate is
> in `docs/BENCHMARKS.md`.

## Why this lane exists (product decision)

| | SGLang lane (sibling) | **EXL3 lane (this)** |
|---|---|---|
| Weights | community NVFP4 | EXL3/TR3 4bpw (byte-identical mirror of brandonmusic) |
| Weight fidelity (KLD vs official) | 0.0605 nats | **0.0246 nats ≈ official FP8, at 54% of the bytes** |
| Runtime | SGLang (6 audited SM121 patches) | vLLM fork + 13-file overlay (sparse-MLA NoPE, fused EXL3 MoE) |
| Validated context | 240k cold (pool ≈ 210k) | **1M** (pool 1.75M tokens at util 0.87) |
| Decode, mono | 37.2 tok/s (DFlash2, standard prompts) | 62.9 structured / 26.9 prose (DFlash2 k=7) |
| TTFT under concurrency | **0.7–1.0 s @ C4** | 6.3–6.6 s @ C4 (prefill-skip policy serializes) |

Read: better weights, longer context, faster high-acceptance decode; worse
batched TTFT. Keep the SGLang lane for bursty agent traffic; use this lane
when weight fidelity or very long context is the priority. See
`docs/QUALITY.md` for the quality evidence.

## Quick start (2× Spark)

```bash
cd vllm-exl3
cp .env.example .env        # edit HEAD_IP / WORKER_IP / NIC names / GIDs
./download.sh               # optional: stage ~164 GiB into the head HF cache
./start.sh                  # pull public GHCR image, sync, launch TP=2
```

API: `http://<HEAD_IP>:8888/v1`, model id `GLM-5.3-Flash-EXL3`.

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5.3-Flash-EXL3",
    "messages": [{"role": "user", "content": "hello!"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

First run copies `.env.example` → `.env`. CLI env beats `.env`
(`SPEC_METHOD=mtp ./start.sh restart` rolls speculation back to MTP k=2).
Needs: passwordless SSH head→worker, docker without sudo on both nodes,
`~180 GiB` free per node. Kit-specific knobs (NIC names, GID indices, memory
fallback) are documented in `.env.example` — read the RoCE block before the
first boot.

## What this vendor changes vs upstream

Serve path (Dockerfile, `overlay/`, inner launch scripts): **upstream,
verbatim.** Host-side bring-up and tooling: hardened.

| Change | Why |
|---|---|
| `HEAD_GID` / `WORKER_GID` per-rank RoCEv2 GID + per-rank preflight | upstream PR #26: some kits need different GID indices per node (e.g. 4/3); a single index kills TP init |
| `HF_BIN` + venv probing + `huggingface_hub` python fallback | upstream issue #22: the `hf` probe died when the CLI lived outside `PATH` |
| Revision-keyed sync marker (`.glm53-exl3-synced`) + `FORCE_SYNC=1` | upstream issue #22: skip the ~164 GiB rsync re-verification on every start |
| Worker-death detection in the `/health` wait (3 strikes ≈ 30 s) | upstream issue #22: fail fast with logs instead of polling for `READY_TIMEOUT` |
| `tests/bench_prefix_cache.py` | cold vs warm follow-up TTFT + prefix-cache hit ratio (upstream's 9.7 s → 1.17 s protocol, made reproducible) |
| `tests/bench_long_context.py` | needle-protocol cold-context bench adapted from the SGLang lane (vLLM `/reset_prefix_cache`, capability gate via `/v1/models`) |
| `tests/run-local.sh` | no-GPU test suite: shell syntax, fragile text contracts, overlay patch tests, bench smoke against a mock server |
| `docs/QUALITY.md`, `docs/KNOWN-ISSUES.md`, `docs/BENCHMARKS.md` | KLD evidence, issue #10/#19 behaviour guides, baseline + promotion criteria |

Upstream scripts keep their names and contracts (`test_start_overrides`,
`test_warm_restart_stdout`, `test_xgrammar_termination` still pass unmodified).

## Benchmarks

Four protocols, all recorded with artifacts in `docs/BENCHMARKS.md`:

```bash
# 1. decode classes + per-position acceptance (requires the lane server)
python3 tests/bench_decode.py --phase structured --structured --runs 5 --max-tokens 400 --skip-coherence --out /tmp/exl3-structured.json
python3 tests/bench_decode.py --phase prose --runs 5 --max-tokens 400 --skip-coherence --out /tmp/exl3-prose.json

# 2. prefix-cache reuse (multi-turn / agentic traffic)
python3 tests/bench_prefix_cache.py --runs 3 --prompt-tokens 7680

# 3. cold long context (needles at 5/50/95%, API health after)
python3 tests/bench_long_context.py --target-tokens 200000 --cold

# 4. concurrency / goodput (repo-level client, runtime-agnostic)
python3 ../bench-glm53.py --model GLM-5.3-Flash-EXL3 --runs 3 --concurrency 4
```

## Known issues you must design around

Read `docs/KNOWN-ISSUES.md` before agent workloads. The two big ones, still
open upstream: **never combine `tools` with `response_format`/`guided_json`**
in one request (xgrammar/GLM47 wire mismatch), and expect cold-prefill queue
stalls (10–40 s) when large contexts land concurrently at
`MAX_NUM_BATCHED_TOKENS=1024`. Blank tool-call arguments can occur under
heavy mixed load (issue #10) — validate client-side and retry.

## Do not

- `--moe-backend marlin`, NVFP4 weights, bf16 or NVFP4 KV (`fp8_ds_mla` is the
  only sparse-MLA path on SM12x)
- pin `TRITON_ATTN` for the drafter (acceptance collapses; see `docs/QUALITY.md`)
- lower `MAX_MODEL_LEN` to "free" KV, or raise `MAX_NUM_BATCHED_TOKENS` to
  "fix" prefix caching
- destroy HF caches, requantize weights, or `docker rm` caches
- change TP, CX7 pins, or `USE_HOST_NCCL` unless re-plumbing NCCL

## Credits & license

Serve recipe, overlay and upstream scripts: **MiaAI-Lab** and contributors
(MIT, retained in `LICENSE`). EXL3 weights: brandonmusic (ShapleyMCG License
1.0), mirrored byte-identically by Mia-AiLab. EXL3 format/kernels:
turboderp (ExLlamaV3). DFlash2 drafter: IncoAI (CC BY-NC-ND 4.0,
research/eval). KLD panel: malaiwah. Issue fixes: upstream PR #26 and issue
#22 authors. Repo-level tooling and this merge: this repo's authors (MIT).
Full credits in `../CREDITS.md`.
