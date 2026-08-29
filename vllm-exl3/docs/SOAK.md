# Multi-day OpenCode soak — protocol (promotion item)

Open promotion item in `docs/BENCHMARKS.md`: *"multi-day OpenCode soak on real
agent traffic, restart included"*. This page is the protocol and the journal.
It exists so the eventual ✅ is an auditable claim, not a vibe.

## What counts

- **Traffic**: day-to-day OpenCode work driven by a human (sessions, subagents,
  tool calls, edits) served by this lane — not synthetic loops. Synthetic load
  is the dedicated `soak_tool_calls.py` harness, used as a *probe*, not a
  substitute.
- **Duration**: at least **3 consecutive calendar days** of real use.
- **Restart included**: at least one planned `./start.sh restart` inside the
  window (warm restart, caches persisted) — a lane that only survives uptime
  has not been validated as an agent runtime.
- **Correctness watch**: zero unrecovered blank/empty required tool-call
  arguments (upstream issue #10). The client-side validation + retry stays
  active all along (`tests/soak_tool_calls.py` is its reference
  implementation; OpenCode applies the same policy on its side).

## Daily procedure (5 minutes)

1. On the head node, run the probe:
   ```
   vllm-exl3/scripts/soak-day.sh
   ```
   It prints a paste-ready markdown block (container uptime + restart counts,
   `/health`, error greps on `logs/head.log` / `logs/worker.log`, and a small
   concurrent tool-call probe) and exits 0 only when everything is healthy.
2. Paste the block into the journal below, oldest last.
3. If the verdict is `UNHEALTHY`, stop and triage before continuing the soak:
   the journal row plus `logs/` decide whether the day restarts the counter.
4. At the end of the window, re-run the two decode classes and one C4 bench
   (`tests/bench_decode.py`, `../bench-glm53.py --concurrency 4`) to confirm
   the perf rows in `docs/BENCHMARKS.md` still reproduce after N days.

## What we are watching for

- crashes / EngineCore death / OOM (the error greps above);
- NCCL watchdog trips or retracts appearing mid-soak;
- TTFT or decode drift vs the recorded rows (boot-to-boot variance is
  expected; monotonic degradation is not);
- blank tool-call arguments (the probe reports `blank_arg_events` — the
  mitigation must keep recovering every one of them);
- KV pool pressure after multi-day sessions (prefix-cache hit ratio in
  `/metrics`, page model still 3584-token aligned).

## Exit checklist (to tick the box)

- [ ] ≥3 daily journal rows below, all `healthy` (or triaged + restarted with
      root cause documented);
- [ ] ≥1 planned restart inside the window, with the following day still
      `healthy`;
- [ ] end-of-window decode + C4 rows within the band of the recorded
      2026-08-29 baselines;
- [ ] zero unrecovered blank tool-call arguments across all probes.

Then add the row to the promotion checklist in `docs/BENCHMARKS.md` and link
this page.

## Journal

> Append blocks here, oldest last, verbatim from `scripts/soak-day.sh`.

<!-- soak journal starts here -->
