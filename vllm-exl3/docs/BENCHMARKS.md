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

## This kit (2× GB10, TP=2, RoCE) — ThinkStation PGX pair, 2026-08-29

Serve path = upstream's, byte-identical (image pulled, no source changes);
host-side hardening only. Field data also filed upstream as MiaAI-Lab issue #32.

| Date | Protocol | Result | Notes | Artifact |
|---|---|---|---|---|
| 2026-08-29 | `bench_decode.py --phase structured --structured --runs 5 --max-tokens 400` | **66.7 tok/s** median (63.3–68.6), TTFT 0.46 s, accept 0.959 / 6.71 per step | **above upstream baseline** (61.7 lab / 62.9 sparkDash, accept 0.918); per-position 1.0/1.0/1.0/0.98/0.97/0.90/0.89 — no late-position collapse, drafter attention path healthy. Serve path identical, so the delta is kit variance + the probabilistic draft sampler, not our changes | `/tmp/exl3-structured.json` |
| 2026-08-29 | `bench_decode.py --phase prose --runs 5 --max-tokens 400` | 25.2 tok/s median (23.6–26.9), TTFT 0.46 s, accept 0.305 / 2.13 | in upstream band (26.9 lab / 27.1 second kit); per-position shape matches the DFlash2 signature — the ~2.6× structured/prose asymmetry is the drafter's known character | `/tmp/exl3-prose.json` |
| 2026-08-29 | `bench_prefix_cache.py` v2.1+ (`--prompt-tokens 8400`) | **hit 0.8541, eff 0.9999** — every full 3584-page of the warm prompt reused; warm TTFT 2.5 s vs 10.3 s true cold (**4.1×**) | page model confirmed: hits are block-aligned to the **3584-token hybrid MLA page** (`floor(tokens/3584)×3584`), and 7168 = 2×3584 exactly. Earlier v1/v2.0 oddities were bench instrumentation, not the stack: a metrics window covering cold+warm halves the ratio, and content reuse across sessions fakes colds because the image has **no cache-reset endpoint** (documented upstream, issue #31; fixed client-side by the per-session salt) | `results/glm53-exl3-prefix-cache-*.json` |
| 2026-08-29 | `../bench-glm53.py --runs 3 --concurrency 4 --thinking off` | 9/9 ok, aggregate **54.8 tok/s**, TTFT median 1.61 s, **p99 16.7 s** | TTFT staircase matches `GLM53_MIXED_PREFILL_CHUNK=skip`: new prefills defer until a decode drains (coding r2 TTFT 9.54 s ≈ coding r1 total 9.50 s). Known upstream tradeoff (issue #19 pattern); candidate softener `GLM53_MIXED_PREFILL_CHUNK=256` | `results/glm53-benchmark-20260829-115750.json` |
| 2026-08-29 | `bench_long_context.py --target-tokens 200000 --cold` | **ok, 3/3 needles (sha256 exact), API healthy** — 200,005 tokens, TTFT 229.5 s, prefill 871.3 tok/s e2e, decode 150.2 tok/s (40-token answer, small sample) | pool 1.75M → 200k ≈ 11%. No reset endpoint on this build → the SESSION filler guarantees cold | `results/glm53-long-context-long-context-20260829-122213.json` |
| 2026-08-29 | `… --target-tokens 500000 --cold --label 500k-cold` | **ok, 3/3 needles, API healthy** — 500,011 tokens, TTFT 598.0 s, prefill 836.1 tok/s | earlier runs' pages still resident; −4% prefill vs 200k | `results/glm53-long-context-500k-cold-20260829-123606.json` |
| 2026-08-29 | `… --target-tokens 900000 --cold --label 900k-cold` | **ok, 3/3 needles, API healthy — 900,007 tokens cold** | TTFT 1138.4 s, prefill 790.6 tok/s; 1.6M cumulative pages resident in the 1.75M pool | `results/glm53-long-context-900k-cold-20260829-125558.json` |
| 2026-08-29 | `… --target-tokens 990000 --cold --label 990k-cold` (after restart) | **ok, 3/3 needles, API healthy — 990,007 tokens: the full 1M window validated cold** | TTFT 1231.9 s, prefill 803.6 tok/s (fresh boot, empty cache); `decode=null` = the minimum-window guard correctly rejected the 40-token sample | `results/glm53-long-context-990k-cold-20260829-133322.json` |

Prefill across the ramp: 871 → 836 → 791 → 804 tok/s — essentially flat
(worst −9%), the sparse-MLA signature.

Still pending on this kit: `coldhit` reading on a salted prefix run,
`GLM53_MIXED_PREFILL_CHUNK=256` C4 comparison, multi-day OpenCode soak.

Promotion criteria (inherited from the sibling lane's DFlash2 protocol, plus
lane-specific items):

- zero crashes/retracts during the sweep; structured and prose both recorded;
  acceptance per position sane (no later-position collapse) ✅ 2026-08-29
- cold long-context 3/3 needles with API healthy after ✅ 2026-08-29 (200k /
  500k / 900k / 990k)
- prefix-cache reuse at or near the page model (eff ≥ 0.9) ✅ 2026-08-29
- `GLM53_MIXED_PREFILL_CHUNK=256` C4 comparison: p99 TTFT meaningfully below
  the `skip` p99 (16.7 s) without giving back the aggregate — decides the
  lane's default scheduler policy ⬜
- tool-calling soak under concurrent load: no blank required arguments
  (upstream issue #10 is open — client-side validation + retry until then) ⬜
- multi-day OpenCode soak on real agent traffic, restart included ⬜
- quality A/B against the official API on identical coding/agent tasks
  (the KLD argument deserves an end-to-end confirmation) ⬜

Flip the repo default (README ordering, `main` merge) only when every box is
ticked; until then this lane is the documented pick for long context and
weight fidelity, and the SGLang lane stays the default for bursty agents.
