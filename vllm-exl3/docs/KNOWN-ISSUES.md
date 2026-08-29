# Known issues & pitfalls (learned from upstream issues + our SGLang lane)

This lane vendors the MiaAI-Lab EXL3 recipe and adds fixes from its public
issue tracker. Read this before production use.

## Fixed in this vendor (vs upstream `main`)

| Upstream ref | Fix |
|---|---|
| [PR #26](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/26) | Per-rank RoCEv2 GID index: `HEAD_GID` / `WORKER_GID` (default = `NCCL_IB_GID_INDEX`). On kits where no single GID index is populated on both devices (observed: 4 on head, 3 on worker), a single index kills TP init with an unhandled NCCL error. Preflight now validates each rank's index and prints both GID tables when refusing. |
| [Issue #22](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/22) (1) | `HF_BIN` override + venv path probing + `python3 -m huggingface_hub.commands.huggingface_cli` fallback, so the download step no longer dies when `hf` lives outside `PATH`. |
| [Issue #22](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/22) (2) | Revision-keyed sync marker on the worker: when the worker cache already matches the head snapshot commit, the ~164 GiB rsync is skipped instead of re-verifying every start. `FORCE_SYNC=1` (or deleting the marker) forces a re-sync. |
| [Issue #22](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/22) (4) | `wait_for_health` detects a dead worker container after 3 consecutive failed inspections (~30 s) and fails with the log dump, instead of polling `/health` for the full `READY_TIMEOUT`. |

## Open upstream — behave accordingly

### Issue #10 — blank/missing required tool-call arguments under concurrent load
Under heavy concurrent load (multi-tool-call turns + large cold prefills),
`tool_calls` sometimes come back with a function name but empty required
arguments. Not reproducible standalone with the same payload — it is a
concurrency/batching-state interaction, still open upstream, and their own
scheduler fix (`GLM53_MIXED_PREFILL_CHUNK=skip`) does not eliminate it.

Mitigations for agent workloads (OpenCode-style):
- keep `MAX_NUM_SEQS` modest (`4`) — the default here;
- validate tool-call arguments client-side and retry the request on empty
  required args (a silent retry reproduced clean output upstream);
- avoid mixing very large cold prefills with interactive tool traffic when
  correctness matters more than throughput.

### Issue #19 — API stalls (10–40 s) under concurrent structured output
Root-caused upstream to (a) the cold-prefill queue at
`MAX_NUM_BATCHED_TOKENS=1024` (the real stall), and (b) an xgrammar/GLM
wire mismatch that only adds log spam and a 2× TTFT penalty:
- **Never send `tools` and `response_format`/`guided_json` in the same
  request.** The GLM47 parser emits XML tool tokens that a JSON grammar
  rejects → `Failed to advance FSM` spam, hallucinated JSON instead of
  `tool_calls`, and acceptance collapse under load.
- Long-conversation turns re-prefill in 1024-token chunks; a ~10k-token cold
  prefix at concurrency 8 measured ~38 s walls. This is capacity, not a bug —
  size expectations accordingly (upstream `tests/repro_grammar_stall.py` has
  the full matrix).
- The xgrammar termination backports (#52805/#53046) shipped in this overlay
  fix the matcher-error paths, not this wire mismatch.

### `TRITON_ATTN` on the drafter — do not pin it
The Triton draft-attention path is causal inside the draft block on this
image; later-position acceptance collapses (0.918 → ~0.31 structured).
SM121 picks `FLASH_ATTN` for the non-causal SWA drafter by default. See
`docs/QUALITY.md`.

## Memory / sizing pitfalls (upstream README, condensed)

- `--kv-cache-dtype fp8` → packed `fp8_ds_mla` is **required**: bf16 KV has no
  sparse-MLA kernel on SM12x; NVFP4 KV does not exist for sparse MLA.
- Do **not** lower `MAX_MODEL_LEN` to "free" KV: logged pool tokens ≈
  concurrency × the cap, and the hybrid (mamba + DFlash window) floor is
  mostly length-independent — shrinking the cap shrinks the pool.
- Keep `SKIP_MM_PROFILING=1` with vision on; the max-size dummy profile OOMs
  this UMA.
- `GPU_MEM_UTIL=0.87` needs ≥105.9 GiB free after vLLM's ~9 GiB init. Nodes
  with resident services: `GPU_MEM_UTIL=0.86` with `MAX_MODEL_LEN=800000`.
