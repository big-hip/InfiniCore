# MTP scope and ordinary inference follow-up

This follow-up keeps the existing vocabulary-parallel head, short verification
Attention, replay-free checkpoints, opt-in draft graphs and exact-prompt snapshots.
It does not remove an existing acceleration feature to reduce the ordinary path's
scope. The feature branch still targets `InfiniLM-v0.2.9c`; the PR remains Draft.

## Code changes

- Select multi-token GDN recurrence only when per-token checkpoints are present.
  Ordinary 2–8-token prompts use the pre-existing chunked path. Remove the now
  unreachable destination fallback and advance the packed-request cursor once.
- Preserve physical KV page zero during ordinary as well as MTP graph recapture.
  This adds temporary capture storage/copies, not per-token replay work.
- `AsyncLLMEngine.stop(timeout=5.0)` reports a timeout without closing an active
  worker's engine. Worker `finally` cleanup closes it after execution returns;
  stopping/closed engines cannot be restarted. Repeated stop is safe.
- Preserve the norm mapping, tied-weight packing and configured FP32 state fixes.
  These intentionally apply to ordinary inference. FP32 state's capacity cost
  remains documented and controlled by `num_state_rows`.
- Add focused checks to existing test files, with no new test modules.

## Validation

- Final TP1 suite: 65 passed, 1 TP2-only skipped (25.29 s).
- Final TP2 suite: 66 passed (31.41 s).
- Includes ordinary Qwen vocabulary projection vs independent unsharded FP32
  projection, MTP enabled/disabled target equivalence, ordinary Qwen2 graph
  recapture, shutdown timeout/in-flight exception and deferred remote KV release.
- Existing K=1/2/4 acceptance, rollback, batching, cancellation and optional
  graph/prefix tests remain enabled and pass. No accuracy tolerance was loosened.
- Build and repository formatting checks pass. InfiniCore is unchanged this round.

## Ordinary-model comparison

Separate builds of upstream `270feb3e` and this feature branch used the same
InfiniCore libraries. Deterministic tiny Qwen2 and Llama FP16 checkpoints (2 layers,
hidden size 256, vocab 512) run eager and ordinary Decode graphs. Prompts contain
4 or 63 tokens, with 16 generated tokens. All compared logits and greedy tokens
are exactly equal; sampled process memory is equal (314 MiB eager / 316 MiB graph).
This isolates LM changes; it is not an end-to-end Core PR comparison.

The initial short latency runs vary substantially. A second run uses a one-second
warmup, five repetitions, one Torch CPU thread and CPU affinity 8–11, with reverse
revision order. The sub-millisecond/millisecond timing remains unstable and is
not accepted as proof of ordinary-model performance non-regression. An A/A repeat of the unchanged current Llama graph binary moves mean Decode
ITL from 0.949/1.107 ms to 0.365/0.447 ms for the same 4/63-token inputs.
Preserve all raw results rather than choosing a favorable run. These synthetic checks establish
numerical/lifecycle compatibility, not production throughput for all models.

## Real Qwen3.8-27B-FP8 MTP comparison

Before: `9f262aa9`. After: `86208ae1`. One RTX A6000 48 GiB, TP1, PP1, K=2,
BF16 activations, E4M3 128x128 block weights via Marlin, eager, greedy, prefix off,
40 KV pages x 64 tokens, 5 recurrent state rows. The same resident model serves
three repeats of each 4/63-token prompt with 32 output tokens, EOS ignored.
Loading and warmup are excluded. Decode rate is 31/(wall time - first-step time).
Other user's process retains GPU contexts (274 MiB on this GPU, observed 0% idle
utilization before the run); hardware was not exclusively reserved. Do not claim
statistical significance or a new speedup from this short sequential comparison.

| Input tokens | Before Decode tok/s | After Decode tok/s | Before TTFT ms | After TTFT ms |
|---|---:|---:|---:|---:|
| 4 | 35.62 | 36.31 | 56.89 | 58.31 |
| 63 | 45.67 | 46.50 | 143.48 | 139.93 |

Values are medians of three repeats. All 192 output tokens match before/after;
acceptance counts match (17/28 for the short prompt, 20/21 for the 63-token prompt).
The short prompt's first step is 1.42 ms slower while its whole request is faster.
Do not claim unchanged Prefill cost for every prompt shape.

Resident process memory: 42404 MiB both versions. Sampled peaks: 43282 vs 43254 MiB
at 500 ms intervals; these are sampled process peaks, not allocator-instrumented
exact allocation peaks. TP2 correctness was rerun; TP2 27B throughput was not.

## Reproduction

Use the same InfiniCore runtime for each LM checkout, with C++11 ABI matching the
runtime (ABI=1 here). Configure/build/install with xmake in separate worktrees.
Set `INFINILM_QWEN_MTP_TEST_MODEL` to the saved tiny FP8 checkpoint and select TP1/2
with `INFINILM_QWEN_MTP_TEST_TP`; execute `python -m pytest test/models/qwen3_5
test/layers/test_pre_transpose.py -q`. The attached scripts reproduce the paired
probes; model paths in scripts must be adapted to the local checkout.
