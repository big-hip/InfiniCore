# Qwen MTP dynamic-batch numerical fix — 2026-09-21

InfiniLM #584's real-model cancellation/re-admission mismatch was reproduced on
the restarted dual RTX 5090 machine, then resolved by aligning short verification
with ordinary Decode arithmetic. The original reproducer now passes without
loosening comparisons, replaying the target, or disabling batching.

## Revisions and conditions

- Before: InfiniLM `86208ae1f200bae769fdf0a07bf2f1e4092fcc2b`.
- After: `78d19f74` on the existing `feat/qwen-greedy-mtp` branch; target remains
  `InfiniLM-v0.2.9c`. Five files, +109/-21; one additional test in an existing
  test module. These research artifacts do not enter the feature PR.
- Same persisted Core runtime/source overlay as the
  [2026-09-20 validation](../qwen-mtp-rtx5090-20260920/README.md). No Core changes
  or rebuild in this fix. Its runtime prerequisites are consolidated in #1566;
  this is not a fresh build of that PR's latest head.
- Two dedicated RTX 5090, 32607 MiB/device, driver 610.43.02, CUDA 13.3,
  PyTorch `2.13.0a0+8145d630e8.nv26.06`, SYS topology without CUDA P2P.
- Qwen/Qwen3.8-27B-FP8, `Qwen3_5ForConditionalGeneration`, dense text, one MTP
  layer, TP2/PP1, greedy, BF16 activations/KV, FP32 recurrent state. FP8 weights
  use **Marlin W8A16**, not native FP8 matrix multiplication. Model provenance,
  prompt IDs and build options are unchanged from the previous validation.
- Lifecycle checks: batch limit 2, 80 pages x 64 tokens, prefix reuse off,
  eager, state rows `1 + 2*(K+2)`, 32 output tokens ignoring EOS.

## Cause and implementation

Two shape-dependent numerical paths separated MTP from ordinary Decode:

1. Packed short verification used Prefill Attention, while the existing
   single-request short path reused Decode. Verification now expands only
   causal lengths and page-table indices per query, using each request's own
   page-table row. All layers share this metadata; no KV payload is copied.
2. Checkpointed GDN verification already projected the small BF16 `a`/`b` gates
   per token. Ordinary batched Decode projected all rows together. Different
   GEMM shapes rounded differently, and recurrence amplified the difference.
   NVIDIA batched Decode now uses the same per-token gate shape. Large quantized
   projections remain batched; ordinary single-token Decode, long Prefill and
   non-NVIDIA ordinary gate dispatch retain their existing behavior.

This **does affect ordinary NVIDIA Qwen batched Decode**, even when MTP is off.
Ordinary output tokens can change from the previous batched baseline. It does
not add state rows, duplicate KV, or claim universal batch-invariant arithmetic
across all kernels, precisions and models.

## Correctness evidence

| Check | Before | After |
|---|---|---|
| Original TP2 K2 fixed packed pair | MTP matches packed ordinary | Same pass |
| Cancel after shared packed Prefill; keep `summary`, admit `code` | Remaining request differs at index 20; new request matches | Both match ordinary token-for-token |
| Same controlled lifecycle with K1 and K4 | Not rerun before | Both pass |
| KV references and recurrent rows after completion/cancel/close | Reclaimed in the failing reproducer | Reclaimed for K1/K2/K4; active/queued close also passes |
| TP2 same-state verification vs ordinary first-token Decode | See TP1 ablation below | 384/384 Conv/GDN tensors and both hidden vectors exactly equal |
| Tiny MTP + ordinary linear-layout regression | Previously 65+skip / 66 | A6000 TP1: 66 passed, 1 TP2-only skip; 5090 TP2: 67 passed |
| Single-request ordinary graph vs K2 | Historical pass | 3 prompts x 3 repeats x 64 tokens, exact outputs |

The same-state diagnostic verifies into scratch rows, then consumes unchanged
committed rows with ordinary Q1 Decode. It compares first-token checkpoints and
hidden states, not two independently evolved trajectories. Tested histories
are 1026/127 tokens with three verification queries per request.

A6000 TP1 real-model ablation at that same point:

| Change | Unequal state tensors / 192 | Hidden relative RMS, request 0 / 1 |
|---|---:|---:|
| Original | 163 | 0.010252 / 0.030386 |
| Packed Decode Attention only | 159 | 0.010041 / 0.029064 |
| Attention + aligned gates | 0 | 0 / 0 |

The new tiny test guards unequal query/history lengths, noncontiguous physical
pages, page-boundary crossing, future-token causality and request isolation.
**It also passes on the old binary**: the real 27B reproducer, not the tiny test
alone, establishes the original numerical failure and its resolution.
TP1 tests preceded the final NVIDIA-only restriction of ordinary gate dispatch;
the NVIDIA branch executed by those tests is unchanged. TP2 tests use final source.

## Performance and memory

Single request, three repeats after warmup, prompts 63/127/1023 tokens, 64 output
tokens ignoring EOS. Median `(N-1)/(wall-TTFT)`, including CPU/scheduler work,
excluding loading and Prefill. Both modes use eager Prefill; ordinary Decode
uses graphs and K2 is eager. These are current implementation results, **not
speedups attributable solely to this numerical fix**.

| Prompt | Ordinary graph tok/s | K2 tok/s | K2 gain |
|---|---:|---:|---:|
| Chinese, 63 | 63.35 | 110.44 | +74.3% |
| Code, 127 | 62.40 | 118.27 | +89.5% |
| Summary, 1023 | 52.00 | 89.67 | +72.4% |

Ordinary batched Decode was checked separately because its gates changed.
Same two prompts (63/1023), 32 outputs/request, B2/TP2, 9 state rows, MTP off,
three repeats after warmup, fixed-work timing, 4 host threads:

| Mode | Before aggregate tok/s | After aggregate tok/s | Change |
|---|---:|---:|---:|
| Eager | 93.50 | 100.04 | +7.0% |
| Decode graph | 92.66 | 99.34 | +7.2% |

Eager/graph outputs agree within each version and all repeats are stable.
Before/after outputs need not agree because the numerical fix intentionally
changes batched arithmetic. These short sequential runs show no regression for
this workload; they do not certify all models, larger batches or traffic mixes.

Device occupancy sampled every 100 ms on idle GPUs, MiB for GPU0/GPU1:

| Workload | Before peak | After peak |
|---|---:|---:|
| Original K2 B2 lifecycle | 24806 / 24810 | 24818 / 24800 |
| Ordinary graph B1 | — | 23768 / 23782 |
| K2 B1 | — | 24384 / 24392 |
| K1 B2 lifecycle | — | 24770 / 24778 |
| K4 B2 lifecycle | — | 25586 / 25586 |

The comparable K2 B2 test peaks at about 24.24 GiB/device in both versions.
K4 intentionally reserves more checkpoint rows. Brief peaks may be missed;
these are device-occupancy samples, not allocator live-byte measurements or a
long-duration leak test. Both GPUs returned to 2 MiB and 0% utilization afterward.

## Reproduction and evidence

`measurements.json` retains original before/after outputs, timings, acceptance,
memory reports and exact-state comparisons; process IDs are removed.
`provenance.json` records source and binary hashes. `checks.txt` contains focused
test output. No model files, binaries, credentials or profiler traces are included.

Reuse the previous validation's `cases.json`, `model_benchmark.py` and
`run_measured.py`. `diagnose_service_controlled.py` is the unchanged reproducer
executed before and after. Scripts expect the staging layout described there.

```bash
source scripts/env.sh
python scripts/diagnose_service_controlled.py --tp 2 --k 2 --output logs/lifecycle.json
# Repeat with --k 1 and --k 4. Original 86208ae1 K2 exits 1; fixed build exits 0.
python scripts/probe_same_state.py --tp 2 --k 2 --require-exact \
  --model models/Qwen3.8-27B-FP8-marlin --cases tools/cases.json \
  --output logs/same-state.json
python scripts/benchmark_batched_decode.py --root /root/infini-mtp-5090 \
  --output logs/ordinary-b2.json
# Repeat with --graph; compare original/fixed native binaries in separate processes.
```

Supported scope remains dense greedy text, one MTP layer, PP1, NVIDIA TP1/TP2.
This resolves the reported controlled dynamic-batch failure; it does not establish
random sampling, multimodal/MoE/PP, multi-layer MTP, other accelerator support,
all possible arrival schedules, or a validated comparison against vLLM MTP.

## CI on the fixed revision

For `78d19f74`, fork [format](https://github.com/big-hip/InfiniLM/actions/runs/35556613781)
and [Ruff](https://github.com/big-hip/InfiniLM/actions/runs/35556612958) passed.
The fork push workflow skips its hardware job; A6000/5090 results above are direct
hardware runs. Upstream [CI](https://github.com/InfiniTensor/InfiniLM/actions/runs/35556617222)
and [Ruff](https://github.com/InfiniTensor/InfiniLM/actions/runs/35556616469) require
maintainer approval (`action_required`). Neither branch was merged.
