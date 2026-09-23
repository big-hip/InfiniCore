# Current-head local GPU regression — 2026-09-23

The current InfiniLM #584 head `31ef429044a114bd8b4bfb1d290edcc06342a2c6` passes its small-model TP1/TP2 regression on two shared NVIDIA RTX A6000 48 GiB cards after rebuilding and installing current InfiniCore #1566 head `a3ac4df44fc902fe223b441e0255631cc6b73a71`. No implementation, test assertion or tolerance was changed during this validation.

## Results

| Suite | Result | Pytest time | Sampled test-process peak |
|---|---|---|---|
| MTP/model contracts + ordinary linear/graph regression, TP1 | 66 passed, 1 TP2-only teardown check skipped | 26.04 s | 370 MiB |
| Same suite, TP2 | 67 passed, no skips | 32.51 s | 468 MiB on each GPU |
| Core graph replay and FP8/BOOL casts | 13 passed | 8.11 s | See summary.json |
| Core paged Prefill, including head size 256 | 88 passed | Not benchmarked | Not sampled separately |

The LM suite totals include CPU checks: they are not counts of GPU-only tests. The fixture is a tiny two-layer Qwen hybrid model (one GDN layer, one full-attention layer, one MTP layer), hidden size 256, head size 128, vocabulary 64, FP32 recurrent state and FP8-block/Marlin weights. This is FP8 storage with BF16 compute, not native W8A8. Core Prefill cases independently cover the head-256 path, FP16/BF16, int32/int64 metadata, GQA/TP-local shapes and MLA.

LM checks include K1/K2/K4 acceptance lengths, continuation after checkpoint selection, state-pool rebuilding, cancellation/re-admission, request/causal isolation, vocabulary boundaries/ties, host/device inputs, ordinary projection and graph recapture. Existing tolerances were retained.

## Environment correction

The first local attempt linked the September 19 Core library and aborted in ordinary graph execution with `Double free detected in PinnableBlockAllocator`. That library predates the allocator/replay fixes already committed in #1566. Rebuilding current Core with CUDA 12.4, SM86, graph support, ATen and the existing CUTLASS headers took 70.574 seconds. After installation, actual loaded library paths were checked; the same isolated graph test passed, followed by the full suites above. This required an environment correction, not a new feature patch.

The local build initially needed its existing `CUTLASS_ROOT` restored. The final build and install succeeded. Source worktrees remain clean. `summary.json` records the tested source revisions plus binary and fixture SHA-256 values.

## Commands

With matching current Core/LM builds and the small checkpoint:

```sh
INFINILM_QWEN_MTP_TEST_MODEL=/path/to/tiny-mtp \
INFINILM_QWEN_MTP_TEST_TP=1 CUDA_VISIBLE_DEVICES=1 python -m pytest \
  test/models/qwen3_5 test/layers/test_pre_transpose.py -q --maxfail=1

INFINILM_QWEN_MTP_TEST_MODEL=/path/to/tiny-mtp \
INFINILM_QWEN_MTP_TEST_TP=2 CUDA_VISIBLE_DEVICES=0,1 python -m pytest \
  test/models/qwen3_5 test/layers/test_pre_transpose.py -q --maxfail=1
```

In the Core checkout, run `python -m pytest test/infinicore/graph/test_sum_scalar_power.py test/infinicore/graph/test_cat.py test/infinicore/ops/fp8_cast.py -q`, then `python test/infinicore/ops/paged_attention_prefill.py --nvidia`.

## Limits

Other users' services remained resident. These are bounded correctness checks, not uncontended throughput/latency measurements. The LM test monitor sampled only its child process, about every 0.5 seconds, with a 4 GiB/device stop threshold and 180-second timeout. Sampling can miss short peaks; neither threshold was reached. Background GPU memory occupancy returned to its observed baseline after the checks.

The full 27B checkpoint could not fit in the remaining shared memory, so its current-head regression is still pending. The tiny fixture does not establish the real model's dynamic-batch numerical behavior across all prompts. Previous 27B results retain their historical source labels. Integration of #573/#575/#584 into one release head is also not covered here.

`test-results.png` renders these saved logs for review; it is not a GitHub CI screenshot. All artifacts live on the separate validation branch, outside the feature PRs.
