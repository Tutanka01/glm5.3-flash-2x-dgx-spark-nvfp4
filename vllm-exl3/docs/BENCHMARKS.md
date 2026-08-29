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

## This kit (2× GB10, TP=2, RoCE) — first hardware results 2026-08-29

Serve path validated on the reference kit (ThinkStation PGX pair, 2026-08-29):
boot clean, DFlash2 k=7, 1M serve. Decode protocol runs below; artifacts were
written to `/tmp` on the kit — copy into `results/` to archive:

```bash
cp /tmp/exl3-structured.json /tmp/exl3-prose.json results/
```

| Date | Protocol | Result | Notes | Artifact |
|---|---|---|---|---|
| 2026-08-29 | `bench_decode.py --phase structured --structured --runs 5 --max-tokens 400` | **66.7 tok/s** median (63.3–68.6), TTFT 0.46 s, accept 0.959 / 6.71 per step | **above upstream baseline** (61.7 lab / 62.9 sparkDash, accept 0.918); per-position 1.0/1.0/1.0/0.98/0.97/0.90/0.89 — no late-position collapse, drafter attention path healthy | `/tmp/exl3-structured.json` |
| 2026-08-29 | `bench_decode.py --phase prose --runs 5 --max-tokens 400` | 25.2 tok/s median (23.6–26.9), TTFT 0.46 s, accept 0.305 / 2.13 | in upstream band (26.9 lab / 27.1 second kit); per-position shape matches the DFlash2 signature | `/tmp/exl3-prose.json` |
| 2026-08-29 | `bench_prefix_cache.py` v1 (--prompt-tokens 7680) | superseded | v1 flaw: sequential runs reused one chat (colds contaminated) and the prompt sat below the 2nd 3584-token page → hit 0.5265 = 3584/6805 exactly. Confirmed the page-granular KpoolTail model; v2 adds unique content, cache reset, hit_efficiency | `results/glm53-exl3-prefix-cache-20260829-112959.json` |
| 2026-08-29 | `bench_prefix_cache.py` v2.1 (--prompt-tokens 8400) | **hit 0.8541, eff 0.9999** — every full 3584-page of the warm prompt reused | page model confirmed (7168 = 2×3584). Open anomaly: cold TTFT dropped 10.3 s → **1.9 s** between sessions (warm steady at ~2.4 s) — impossible for a full 8359-token prefill at the ~810 tok/s rate; `coldhit` instrumentation (v2.2) + server `StartedAt` will discriminate prefill-rate-change vs phantom cold hits | `results/glm53-exl3-prefix-cache-20260829-115706.json` |
| 2026-08-29 | `../bench-glm53.py --runs 3 --concurrency 4 --thinking off` | 9/9 ok, aggregate **54.8 tok/s**, TTFT median 1.61 s, **p99 16.7 s** | TTFT staircase matches `GLM53_MIXED_PREFILL_CHUNK=skip`: new prefills defer until a decode drains (coding r2 TTFT 9.54 s ≈ coding r1 total 9.50 s). Known upstream issue #19 tradeoff; candidate softener `GLM53_MIXED_PREFILL_CHUNK=256` (restart) | `results/glm53-benchmark-20260829-115750.json` |

Prefix-cache page model (confirmed by the v1 run): hits are block-aligned to
the **3584-token hybrid MLA page** — `floor(tokens/3584)×3584` at best.
Upstream's 7168/10752/14336 hits are 2/3/4 pages. Rerun with v2:

```bash
python3 tests/bench_prefix_cache.py --runs 3            # default ~8.4k target → 2 pages
```

| 2026-08-29 | `bench_long_context.py --target-tokens 200000 --cold` | **ok=True, 3/3 aiguilles (sha256 exact), API saine** — 200 005 prompt tokens, TTFT 229.5 s, prefill 871.3 tok/s e2e, décode 150.2 tok/s (réponse de 40 tokens, petit échantillon) | capacité : pool 1.75M → 200k ≈ 11 % ; pas d'endpoint de reset sur ce build → le filler SESSION garantit le froid ; vs lane SGLang : prefill 1230 tok/s mais plafond pool ~210k, décode 39.6 tok/s | `results/glm53-long-context-long-context-20260829-122213.json` |
| 2026-08-29 | `bench_long_context.py --target-tokens 500000 --cold --label 500k-cold` | **ok=True, 3/3 aiguilles (sha256 exact), API saine** — 500 011 tokens, TTFT 598.0 s, prefill 836.1 tok/s | pages résiduelles des runs précédents toujours dans le pool ; dégradation prefill 200k→500k : −4 % | `results/glm53-long-context-500k-cold-20260829-123606.json` |
| 2026-08-29 | `bench_long_context.py --target-tokens 900000 --cold --label 900k-cold` | **ok=True, 3/3 aiguilles (sha256 exact), API saine — 900 007 tokens validés à froid** | TTFT 1138.4 s, prefill 790.6 tok/s (−9 % vs 200k — dégradation quasi plate, signature sparse-MLA) ; pages 200k+500k encore résidentes (1.6M cumulés dans le pool 1.75M) ; `decode=516638 tok/s` dans l'artefact = artefact de fenêtre (garde minimum ajoutée au bench) | `results/glm53-long-context-900k-cold-20260829-125558.json` |

| 2026-08-29 | `bench_long_context.py --target-tokens 990000 --cold --label 990k-cold` (après restart) | **ok=True, 3/3 aiguilles (sha256 exact), API saine — 990 007 tokens : fenêtre 1M validée à froid** | TTFT 1231.9 s, prefill 803.6 tok/s (boot neuf, cache vide) ; `decode=null` = la garde de fenêtre a bien filtré l'échantillon 40 tokens | `results/glm53-long-context-990k-cold-20260829-133322.json` |

Still pending on this kit: prefix-cache v2.2 `coldhit` reading, boot shape-warmup check.

Promotion criteria (inherited from the sibling lane's DFlash2 protocol):
zero crashes/retracts during the sweep; structured and prose both recorded;
acceptance per position sane (no later-position collapse); cold long-context
3/3 needles with API healthy after; p99 TTFT at C4 within +10% of the
baseline run before adopting for production.
