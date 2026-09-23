# Remove optional MTP draft graphs

InfiniLM PR #584, `70db65c2` -> `dc431b9be9cd1bf1a752743b216fb2400a15830a`.
The original source is retained locally as `archive/mtp-draft-graph-20260923`
and in the existing public commit history. No history was rewritten.

## Decision and effect

The archived paired RTX 5090 comparison did not establish a benefit from K1
draft graphs: eager/graph rates were 89.40/89.13, 92.20/92.09 and 78.93/78.26
token/s. These sub-1% differences do not establish a meaningful slowdown either.
K2/K4 already use eager and do not use the removed graphs. See the
[original conditions and results](https://github.com/big-hip/InfiniCore/tree/38cd1a290a65ba2dfe49b2476657aa5f371aebfa/docs/validation/qwen-mtp-rtx5090-20260920).

Remove draft capture for one/two input tokens, its map and verification metadata,
hidden-state graph output ownership/copies, and draft replay dispatch. Restore
the ordinary Decode capture loop, sharing the state-protection implementation
already used by the Mamba-2 PR. Remove the new Python `compile()` binding; retained
ordinary graph recapture checks use existing `process_weights_after_loading()`.

Both EngineConfig and native engine construction reject built-in MTP plus graph
compiling. K1/K2/K4 MTP use eager. Ordinary inference still supports Decode graphs.
No model math, FP8 packing, vocabulary projection, acceptance/checkpoint algorithm,
prefix snapshot implementation or scheduling policy changed.

The 11-file cleanup removes 77 net production lines and one documentation line;
focused tests add 19 net lines, for a total reduction of 59. The entire PR now
changes 51 files instead of 52, because `paged_compiler.hpp` returns to upstream.

## Verification at the removal commit

- Rebuilt and installed the native LM extension against the existing matching Core
  runtime: passed (incremental build, 58.242 s; not a new full Core build).
- CPU scheduling, lifecycle and weight-remapping suite: **51 passed, 9 GPU-fixture
  checks skipped** (5.36 s).
- Native MTP+graph rejection using the real model's configuration on CPU:
  **1 passed, 11 deselected** (5.09 s). No model weights or GPU kernels are needed
  for this pre-worker validation.
- Repository formatting (clang-format 21.1.8, Ruff 0.15.20) and diff checks passed.
- The new extension imports and no longer exposes the optional Python `compile()`.

Fresh GPU regression is **pending**: the remote 5090 accepted TCP/SSH but rejected
the existing authentication, and local A6000s have other users' tasks resident.
No other user's process was stopped and no GPU workload was launched.
The earlier GPU/performance matrix is historical and is not reported as having
passed on this removal commit.

The retained GPU suite covers K1/K2/K4 eager equivalence, accepted-state selection,
cache reset, packed/request isolation, cancellation, prefix snapshots, ordinary
hybrid-model graph recapture and ordinary dense-model packing/recapture. When an
idle GPU environment is available, rebuild this head and run:

```sh
INFINILM_QWEN_MTP_TEST_MODEL=/path/to/tiny-qwen-mtp \
INFINILM_QWEN_MTP_TEST_TP=1 python -m pytest \
  test/models/qwen3_5 test/layers/test_pre_transpose.py -q
# Repeat with two visible GPUs and INFINILM_QWEN_MTP_TEST_TP=2.
```

## Review of the other open feature PRs

Counts below are added/deleted implementation lines relative to the requested
release base, excluding Markdown and test/validation directories. They include
user-facing conversion/benchmark entrypoints, not just device kernels.

| PR | Implementation additions/deletions | Decision |
|---|---:|---|
| LM #573 | +469/-147 | Retain cache policy, chunk scheduling and native output/lifecycle contracts; measured work/wait reductions justify them. |
| LM #575 | +704/-32 | Retain model loading, Mamba-2 execution, converter and state/capture fixes; they enable deployment. |
| LM #584 after cleanup | +2063/-161 | Remove optional draft graphs; retain core MTP and the supporting FP8/loading/state/service paths. |
| Core #1558 | +85/-46 | Retain linked-library ABI detection; required for affected MetaX imports. |
| Core #1562 | +655/-2 | Retain device scan and MetaX precision control; required for Mamba-2 execution/validation. |
| Core #1566 | +97/-41 | Retain scalar conversions, communicator cleanup, head-256 dispatch and graph correctness fixes. |

Exact-prompt snapshots remain opt-in with a zero default budget. The historical
A6000 service check showed 1873.4 ms for a 1023-token cold prompt including save,
versus 7.21 ms for a complete hit, using 214.82 MiB of snapshot tensors. That is a
single scenario, not a general serving benchmark, but supports a specific use.
FP8 compatibility dequantization is a slower device fallback; performance runs
must explicitly select the existing Marlin path on NVIDIA. Neither is removed
solely because it is optional or slower in a different workload.

The other five feature PRs were not modified in this cleanup. Core #1566's stale
description of the previously fixed dynamic-batch mismatch was corrected to
**resolved** separately; its production implementation is unchanged.
