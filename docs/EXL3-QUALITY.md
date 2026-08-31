# Quality: EXL3 4bpw vs the alternatives

The lane exists because of this table. Weights decide answer quality; the
runtime decides speed and context. This page records the independent quality
evidence for the checkpoint this lane serves, and the protocol to extend it.

## Teacher-logit KLD panel (independent, malaiwah)

KLD(teacher ‖ model) against the official BF16/FP8 logits, five cold runs,
25 sealed windows (51,175 positions), published on the
[4bpw discussion](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/discussions/1#6a9144846b0bdba943bfe86f).
It scores **the weights**, not a runtime — any stack serving these bytes gets
this quality.

| Checkpoint | Mean KLD (nats) | Size | Reading |
|---|---:|---:|---|
| TR3 K6 (6bpw) | 0.013723 | 254 GB | overkill for this lane |
| Official FP8 (cross-stack) | 0.020615 | 328 GB | reference |
| **EXL3 4bpw (this lane)** | **0.024555** | 176 GB | **≈ official FP8 at 54% of the bytes** |
| Official FP8 (brandonmusic stack, v44) | 0.024629 | 328 GB | same-stack reference |
| NVFP4 (brandonmusic stack, v44) | 0.060535 | ~180 GB | ~2.5× the drift of 4bpw EXL3 |

Why it matters here: the sibling SGLang lane in this repo serves a community
**NVFP4** checkpoint at a similar footprint. Same bytes-per-weight class,
measurably different fidelity. Technical validation (cold-context runs, needle
retrieval) says nothing about this axis — that lane's README says so itself.

## Speculative-decode acceptance is a quality-adjacent signal

DFlash2 k=7 acceptance distribution per position (upstream kit, temp 0,
thinking off, 400 tokens, median of 5):

| Prompt class | pos0..6 accept | accept/step |
|---|---|---:|
| Structured (count 1→200) | 0.98 / 0.98 / 0.94 / 0.94 / 0.91 / 0.83 / 0.83 | 0.918 (6.43) |
| Prose (hash-map explanation) | 0.75 / 0.58 / 0.41 / 0.28 / 0.16 / 0.09 / 0.06 | 0.332 (2.33) |

A collapse at later positions while pos0 stays healthy is the signature of an
attention-mask/backend bug, not of bad weights: upstream measured pinning
`TRITON_ATTN` on the drafter drops structured acceptance to ~0.31 because that
backend masks causal *inside* the draft block. Use `tests/bench_decode.py
--phase structured --structured` and `--phase prose` to reproduce both
regimes; never quote the structured number without the prose number next to
it.

## Reproduce / extend on this kit

Commands run from `vllm-exl3/` unless noted.

1. Serve the lane (`./start.sh`), then run both decode classes:
   ```
   python3 tests/bench_decode.py --phase structured --structured --runs 5 --max-tokens 400 --skip-coherence --out /tmp/exl3-structured.json
   python3 tests/bench_decode.py --phase prose --runs 5 --max-tokens 400 --skip-coherence --out /tmp/exl3-prose.json
   ```
   Record `tok_s_median`, `accept_ratio_median`, and the per-position `pos[]`
   table into the root journal [BENCHMARKS.md](BENCHMARKS.md).
2. For the end-to-end quality A/B (promotion item — closer to real usage than
   KLD): identical coding tasks, temperature and token budgets on both sides,
   with raw completions captured and graded deterministically:
   ```
   export ZAI_API_KEY=...
   python3 ../bench-glm53.py \
       --prompts tests/ab_quality_prompts.jsonl \
       --runs 1 --thinking off --save-content \
       --compare-base-url <official API> --compare-model glm-5.3-flash \
       --output results/ab-quality-<stamp>.json
   python3 tests/grade_ab_quality.py --artifact results/ab-quality-<stamp>.json
   ```
   The grader exec-checks the two coding tasks against fixed unit tests and
   structurally checks the two JSON tasks; a task passes only if every run
   passed. Record both PASS RATEs (EXL3 lane vs official API) in
   `BENCHMARKS.md` — an A/B without the grader output is an opinion.
3. When this kit's numbers exist, add them to `BENCHMARKS.md` with the
   artifact name — never replace upstream's baseline rows.
