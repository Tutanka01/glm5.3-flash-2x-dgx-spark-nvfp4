# Benchmarks — GLM-5.3-Flash EXL3 lane

Rules (inherited from the repo's benchmarking culture):

- Record the protocol with every number: prompt class, runs, tokens,
  temperature, thinking on/off.
- Upstream numbers score **their kit**; they are the baseline to beat or
  reproduce, not a claim about this kit.
- A row for this kit is only valid with its JSON artifact in `results/`
  (gitignored — keep the filename in the row).
- Do not quote a structured-output tok/s without the prose tok/s next to it
  (acceptance regime differs ~2.8×; see `docs/QUALITY.md`).

## Upstream baseline (MiaAI-Lab kit, 2026-08-28, DFlash2 k=7, temp 0, thinking off, 400 tok, CUDA graphs)

Decode, sparkDash protocol — Structured = count 1→200 (high-accept), Code =
clamp prompts. Stream tok/s is per request; aggregate is all streams.

| Concurrency | TTFT | Stream tok/s | Aggregate tok/s |
|---:|---:|---:|---:|
| ×1 | 719 ms | 62.9 (structured) | 62.9 |
| ×2 | 6.62 s | 51.7 | 103.3 |
| ×4 | 6.30 s | 37.1 | 146.5 |

Lab `tests/bench_decode.py`, same protocol, median of 5 × 400, C1:
structured **61.7** tok/s (0.918 accept / 6.43 per step); prose **26.9**
(0.332 / 2.33). Long context / mixed (~60–100k KV) 24–27. MTP k=2 baseline
~24.6.

Prefix caching (1M serve, real user + assistant + follow-up):

| Turn | Hits | Prompt tok | TTFT |
|---|---:|---:|---:|
| ~7.7k cold | 0 | 7696 | 9.7 s |
| ~7.7k follow-up | 7168 (93%) | 7717 | **1.17 s** |
| ~12k follow-up | 10752 | 12015 | 1.94 s |
| ~16k follow-up | 14336 | 16015 | 2.18 s |
| 4× ~7.5k concurrent follow-ups | 7168 each | 7515 each | 1.86–2.50 s |

Context capacity: 1M `max_model_len` with pool 1,754,237 tokens (1.75×) at
util 0.87; live ~256k ×3 concurrent held (29.5% peak KV).

## This kit (2× GB10, TP=2, RoCE) — to be filled on first hardware validation

Status: **not yet validated**. The serve path is upstream's (reproduced on a
second kit upstream, 2026-08-28: structured 38–62, prose 27.1 after
kit-specific GID/NIC adjustments), but this vendor adds host-side changes
(sync marker, health-watch, per-rank GID) that only touch bring-up, not the
serve path. First validation must cover:

1. Boot + smoke: `./start.sh`, then `curl :8888/v1/models` →
   `GLM-5.3-Flash-EXL3`, one temp-0 generation.
2. Decode classes: `tests/bench_decode.py --phase structured --structured
   --runs 5 --max-tokens 400 --skip-coherence` and `--phase prose --runs 5` →
   record `tok_s_median`, `accept_ratio_median`, per-position table.
3. Prefix cache: `python3 tests/bench_prefix_cache.py --runs 3
   --prompt-tokens 7680` → cold vs warm TTFT + hit ratio (expect the
   upstream 93%-reuse / ~8× TTFT regime, modulo scheduler).
4. Long context: `python3 tests/bench_long_context.py --target-tokens 200000
   --cold` then probe toward 1M (needle protocol, 3/3 retrieval + API health).
5. Concurrency: `../bench-glm53.py --runs 3 --concurrency 4` (runtime-agnostic
   OpenAI client: TTFT p99, aggregate goodput).

| Date | Protocol | Result | Artifact |
|---|---|---|---|
| — | — | — | — |

Promotion criteria (inherited from the sibling lane's DFlash2 protocol):
zero crashes/retracts during the sweep; structured and prose both recorded;
acceptance per position sane (no later-position collapse); cold long-context
3/3 needles with API healthy after; p99 TTFT at C4 within +10% of the
baseline run before adopting for production.
