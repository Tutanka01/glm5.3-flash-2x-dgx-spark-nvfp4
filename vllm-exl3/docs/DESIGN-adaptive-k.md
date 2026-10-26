# DESIGN — adaptive verification length for DFlash2 (default OFF)

Status: proposed, not wired by default.
Reference: Reederey87 kit (Apache-2.0), `overlay/patch_adaptive_k.py`.
Scope: `vllm-exl3`, TP2, GLM-5.3-Flash EXL3 4bpw, DFlash2 k=7.

## The one rule

Trim the **target verification prefix**, never the **drafter execution**.

On this V2 stack, `num_spec_tokens_to_schedule` sizes the DFlash2 drafter
(`num_query_per_req = 1 + k`, grouped conv, selector walk, draft-logit cache).
The target verify length follows `len(request.spec_token_ids)`. Writing the
drafter count is a trained-contract change and is a correctness bug for a
non-causal drafter. The overlay therefore:

1. observes accepted drafts in `update_from_output`;
2. trims only `request.spec_token_ids` in `update_draft_token_ids`;
3. captures extra **target** graphs for verify lens 3 and 5;
4. refuses any `batch_k()` hook into `num_spec_tokens_to_schedule`.

## Policy

`_Glm53AdaptiveK` is CPU-only:

- state: `request_id -> [ema, observed_steps]`;
- observation: `obs = accepted` (or `k_max` when fully accepted and
  `SATURATE=max`); `ema = obs*alpha + ema*(1-alpha)`;
- choice: unseen or `< MIN_STEPS` → full k; otherwise
  `target = ceil(ema + margin)`, `n = max(SET value <= min(target, k))`;
- apply: if every request has a choice, trim the uniform row to the min; else
  keep full k unless `BATCH_MARGIN=1` (then trim to the min known choice);
- structured requests stay at full k and pin the batch;
- `POS_TRIM=1` additionally clamps to `min(SET)` only when 2+ observed
  requests share the batch and every request has a choice (C1 no-op).

Defaults: `SET=2,4,7`, `ALPHA=0.25`, `MARGIN=1.0`, `MIN_STEPS=4`,
`SATURATE=max`, `BATCH_MARGIN=0`, `POS_TRIM=0`.

## Anchors

Scheduler (`v1/core/sched/scheduler.py`):

- `import itertools\nimport time\n` — add `import os` if absent;
- `from vllm.compilation.cuda_graph import CUDAGraphStat` — helper insert;
- `update_from_output`: the `num_draft_tokens` / `num_sampled` /
  `num_accepted` block — append `observe()`;
- `update_draft_token_ids`: full body — collect `(request, structured)` and
  call `apply()`;
- the dynamic-SD block — verify only, never rewrite;
- fail closed if a draft `batch_k()` hook is present.

CUDA graphs (`v1/worker/gpu/cudagraph_utils.py`):

- `@dataclass(frozen=True)\nclass BatchExecutionDescriptor:` — helper insert;
- `else:\n    decode_query_lens = [self.decode_query_len]` — append the
  extra-lens helper;
- extra lens are `k+1` for `k in SET` (3 and 5; 8 is native).

The user's image shares the Reederey base digest and only
`patch_scheduler_decode_floor.py` touches `scheduler.py`, so the anchors are
expected present. Still preflight on copies extracted from the built image.

## Capture sizes

Stock DFlash2 list: `1 2 4 8 16 24 32` (C4 × q8).

At C4 with `SET=2,4,7`, the union is
`1 2 3 4 5 6 8 9 10 12 15 16 20 24 32` (15 sizes). At C6 add 18/25/30 and the
full-k 48; at C8 add 21/35/40 and 56/64. Extra graphs cost ~1.1–1.2 GiB/rank
at C4; do not enable C6/C8 capture without a MemFree measurement.

## Wiring (when adopted)

- new `overlay/patch_align_floor.py` (dormant at LPTT < 3584; default on);
- new `overlay/patch_adaptive_k.py` (default off);
- `start.sh`: host vars, preflight, worker scp, both mounts, both rank
  scripts, env forwarding, capture-list generator;
- ordering: decode-floor → align-floor → adaptive-k;
- env: `GLM53_ADAPTIVE_K=off|ema|on|1`, `_CAPTURE=0|1`,
  `_SET=2,4,7`, `_ALPHA`, `_MARGIN`, `_MIN_STEPS`, `_SATURATE`,
  `_BATCH_MARGIN`, `_POS_TRIM`, `_CAPTURE_MAX_BATCH`;
- default-off must emit the stock capture list byte-identically.

## Measurement protocol

A → B0 → B, same boot class, same prompts, temperature 0, thinking off, 5 runs:

| Cell | Policy | Capture | Graphs |
|---|---|---|---|
| A | off | 0 | stock |
| B0 | off | 1 | extra 3/5 |
| B | ema | 1 | extra 3/5 |

Adopt only from **B vs B0** (the graph cost is in B0). Record median tok/s,
accept ratio, accepted/step, chosen-length histogram, TTFT, p50/p99
inter-token gap, MemFree, pool, NaN/Xid/IMA. Return-A rollback with the two
env lines.

## Risks

- **Async scheduling**: Reederey runs `--no-async-scheduling`; the user's
  `start.sh` does not pass it. Assert the boot config is sync before enabling;
  otherwise leave the policy off.
- **Structured pinning**: `apply()` uses
  `structured_output_manager.should_advance(request)`, while `batch_k()` uses
  `request.use_structured_output`. Pin on the latter too; add a
  `should_advance=False` test.
- **Fixed 8-row draft**: the drafter still proposes 7; only target verify
  rows shrink. Gains are smaller if draft TP=2 is not the bottleneck.
- **Memory**: extra full graphs at C4 ~1.1–1.2 GiB/rank; C6/C8 multiply.
- **Losslessness**: no committed token is dropped, but reduction order and
  batch composition change; gate on semantics/acceptance, not byte identity.

## Definition of done

- Host tests ported (`test_adaptive_k_patch.py`, `test_adaptive_k_wiring.py`,
  `test_align_floor.py`) + compose/fail-closed/structured-pin tests.
- `bash -n start.sh`; both ranks log the extra lens; rollback proven.
- Cluster B vs B0: prose/agentic ≥ +5 % or the lane's declared bar,
  structured ≥ 97 % of B0, acceptance 7/7, no NaN/Xid/IMA, p99 gap not worse,
  pool unchanged, MemFree ≥ 2.5 GiB.
- If the bar is missed, leave the overlay default OFF and record the result.
