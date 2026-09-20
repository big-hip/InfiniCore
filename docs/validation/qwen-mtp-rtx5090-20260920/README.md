# Qwen MTP on two RTX 5090 GPUs — 2026-09-20

Single-request greedy generation passed exact output checks and showed useful
MTP throughput gains. **Dynamic cancellation/re-admission did not pass exact
output equivalence.** Resource reclamation passed; the numerical failure remains
unresolved. This supplements the PRs without changing their production code.

## Revisions and conditions

- InfiniLM #584: `86208ae1f200bae769fdf0a07bf2f1e4092fcc2b`.
- InfiniCore #1566: `101e6af45f45e02c8bd83465805394fa2c90a56b` plus the recorded
  `core-5090-runtime.patch`: graph production changes and existing memory-test
  changes from #1560 (`2c342137b61a68463932c07eebb09ce6896a16cf`) and SM120 build
  enumeration. Newly added #1560 graph tests are not in this overlay. Neither
  feature PR head changed; the unmodified release was not used for integration.
- Exact source archives and pinned submodules; CUTLASS
  `087c84df83d254b5fb295a7a408f1a1d554085cf`. NVIDIA/ATen/CCL/graph build, SM120,
  C++11 ABI=1. Two RTX 5090, 32607 MiB/device; driver 610.43.02, CUDA 13.3.
- SYS topology across NUMA nodes, CUDA P2P unavailable. vLLM disables its custom
  allreduce and uses PyNCCL. Host RAM 96 GiB, CPU quota 50 cores. No CPU pinning.
- PyTorch `2.13.0a0+8145d630e8.nv26.06`, vLLM
  `0.22.1+7b9cb5b7.nv26.6.55098374`, Transformers 5.6.0, NCCL 2.30.5,
  Triton 3.6 development build; Nsight Systems 2026.5.1.
- All 74 model files, 30,889,968,033 bytes (28.77 GiB), verified by size/SHA256.
  ModelScope `Qwen/Qwen3.8-27B-FP8`, architecture `Qwen3_5ForConditionalGeneration`,
  64 layers, hidden size 5120, head size 256, 24 query/4 KV heads, one MTP layer,
  vocabulary 248320. The manifest identifies every downloaded revision.
- InfiniLM uses **FP8 weights with BF16 computation (Marlin W8A16)**. Native
  W8A8 was evaluated through vLLM kernels, not integrated into InfiniLM.
  A separate config view symlinks weights and adds
  `quantization_config.fp8_backend="marlin"`.

## Performance and memory

TP2/PP1, text, greedy, batch=1, prefix reuse off. Identical pre-tokenized cases;
fixed output length, ignoring EOS only for these throughput checks. An 8-token
warmup and a 2-token long-input warmup precede three repeats per case.
Rate = `(output_tokens-1)/(request_time-TTFT)`, including scheduler/CPU work,
excluding loading and Prefill. Values are medians, not service percentiles.

InfiniLM uses 80 pages x 64 tokens, BF16 KV/activations, FP32 recurrent states,
2 state rows for ordinary and K+3 for MTP, PyTorch/OMP threads=4. K2/K4 target
verification and draft are eager; all prompt Prefill is eager.

| Mode | Chinese 63→64 tok/s | Code 127→64 tok/s | Summary 1023→64 tok/s | Peak MiB GPU0 / GPU1 |
|---|---:|---:|---:|---:|
| InfiniLM ordinary eager | 62.91 | 62.80 | 52.44 | 23774 / 23776 |
| InfiniLM ordinary Decode graph | 63.21 | 62.43 | 52.16 | 23766 / 23774 |
| InfiniLM K1 eager | 89.40 | 92.20 | 78.93 | 24396 / 24394 |
| InfiniLM K1 draft graph | 89.13 | 92.09 | 78.26 | 24440 / 24440 |
| InfiniLM K2 eager | 107.25 | 116.50 | 86.79 | 24394 / 24398 |
| InfiniLM K4 eager | 114.69 | 143.02 | 88.17 | 24774 / 24766 |
| vLLM ordinary graph, native FP8 | 73.30 | 73.29 | 73.40 | 27104 / 27104 |

K2 gains 69.7%, 86.6%, 66.4% against ordinary graph. Ordinary eager is slightly
faster for code/summary: against the faster ordinary mode per case, K2 gains
69.7%, 85.5%, 65.5%. Graph/eager differences below 1% do not establish meaningful
graph gains; K1 draft graph also shows no consistent benefit. All InfiniLM
single-request outputs match ordinary graph token-for-token; repeats are stable.
K2 accepted/proposed counts per request: 40/46, 41/42, 35/56; K4: 45/68, 49/54,
41/85. More candidates do not guarantee proportional speedup.

Three-repeat longer checks also match exactly: math 256→128 gives
60.29→112.23 tok/s (ordinary graph→K4); English 2047→128 gives 43.63→94.92.
Combined two-case process peaks are 23974 MiB/device ordinary and 25012 K4.
Raw per-run times, ranges, acceptance and output IDs accompany this report.

Memory is total device occupancy sampled every 100 ms on dedicated idle GPUs,
including workers; short peaks can be missed. K2 steady peak is 24340 MiB/device.
Loading/capture, warmup and steady phases are recorded separately. These are not
allocator live/reserved counters. vLLM uses GPU budget 0.85, max length 4096,
max sequences 1, max batched tokens 4096, native SM120 CUTLASS/FlashAttention and
default compile/full-and-piecewise graphs. Its hybrid cache allocation differs
substantially from InfiniLM; occupied memory is not a matched-capacity comparison.
First compile takes about 152 s, initial readiness about 261 s, excluded from rates.
Cross-framework token equality is not an acceptance requirement.

## Correctness and graph verification

| Check | Result / boundary |
|---|---|
| Core E4M3/BOOL casts | 2 passed |
| Core paged Prefill | 88 passed, including existing dtype/metadata/MLA coverage |
| Existing Core paged Decode C API suite | Passed |
| Tiny Qwen MTP TP1 | 65 passed, 1 TP2-only communicator test skipped |
| Tiny Qwen MTP TP2 | 66 passed |
| Tiny Qwen2/Llama ordinary inference | Each 6 requests x 16 tokens; eager/graph IDs equal |
| Real 27B K1/K2/K4 single request | Exact tokens, stable acceptance, KV/state reclaimed |
| Real K1 graph, three captures during active requests | All 32-token continuations exact |
| Real K1 graph, cache/state pool rebuild | Generation counter changes; 32 subsequent tokens exact |
| Fixed packed MTP | Matches packed ordinary; packed ordinary differs from serial |
| Cancel/re-admit, controlled initial packed Prefill | **Exact output failure**; cancellation/close cleanup passes |

Recapture/rebuild peak: 24382/24380 MiB; steady/recapture: 24164/24156 MiB.
No observed growth above initial load peak in these three repetitions. This is
not a long-duration leak test or K2/batched graph coverage. Nsight records
**62 `cudaGraphLaunch` calls**, 31 ordinary Decode steps x two ranks, confirming
actual graph replay rather than merely a configuration flag.

An extra head-256 FP32-oracle probe passes contiguous and strided/multi-request
cases, causality and ALiBi. Forced old `ref` dispatch fails the combined
strided/multi-request/ALiBi case for BF16/FP16; the triggering feature is not
isolated. Failures are retained. Three-repeat operator timings are exploratory,
not an attribution of the whole-model MTP speedup.

## Unresolved failures

**InfiniLM:** the first packed comparison differs from serial ordinary at output
index 20 (21st token) for `summary`. Packed ordinary also differs there; fixed
packed MTP matches packed ordinary. A controlled experiment starts both modes
with identical packed Prefill and one generated token, cancels `zh`, admits
`code`: `code` matches, remaining `summary` MTP differs from ordinary at index 20.
All KV references and Conv/GDN rows are released; cancel status and close with
active/queued requests pass. Cleanup success does not imply output equivalence.

At shared accepted prefix position 1042, FP64 dot products using the BF16 hidden
vector and BF16 head weights produce these scores (diagnostic recomputations,
not a claim about exact internal GEMM accumulators):

| Path | Token 96746 | Token 114734 |
|---|---:|---:|
| Packed ordinary | 19.6842403 | 19.7279905 |
| Packed MTP | 19.6651176 | 19.6890030 |
| Cancel/re-admit ordinary | 19.6842403 | 19.7279905 |
| Cancel/re-admit MTP | 19.6905426 | 19.6844617 |

Ranking flips; hidden relative RMS ordinary vs cancel-MTP is 0.01047, max absolute
difference 0.140625. This identifies decision sensitivity, **not the first
divergent layer**, and does not rule out all state bugs. Tolerances stay unchanged;
reproduction exits 1. General 5090 dynamic-batching acceptance requires resolving
this or explicitly narrowing support.

**vLLM K2:** the first greedy token differs from its ordinary baseline
(109727 vs 96212). Strict timing stops after that request. Observed 143.66 tok/s
is one failed-validation diagnostic, not a validated median/speedup. A separate
run profiles with vLLM's Torch profiler after the timed request and retains
`success=false`, exit 1. Root cause is not established. No claim of outperforming
validated vLLM MTP is made.

## Profiling and native FP8

Nsight uses separate 1023-prompt/32-output runs, excluded from throughput. Kernel
sums are per device; do not sum across GPUs or add overlapping CPU waits.

- K2 initial Prefill wall 1369.99 ms: GDN chunk Prefill 1139.87/1132.50 ms per GPU
  (about **83%**), Marlin 113.37/112.69 ms, Attention 17.10/16.91 ms,
  NCCL 54.28/62.55 ms. Qwen GDN Prefill is the primary bottleneck.
- Twelve subsequent MTP steps wall 448.60 ms with profiling: Marlin about
  105 ms/GPU, Attention 49.3 ms/GPU, GDN recurrence 5.83 ms/GPU. NCCL is
  31.30 ms rank0 vs 162.27 ms rank1; busy interval unions 251.11/382.14 ms.
  Investigate submission/wait imbalance; NCCL duration includes rank waiting,
  not only data transfer.
- Ordinary graph TTFT medians 98.38/185.76/1363.87 ms; vLLM ordinary
  63.19/67.73/155.85 ms. MTP does not remove target Prompt Prefill latency.
- vLLM's diagnostic profile uses a different 63/16 workload. SM120 CUTLASS,
  FlashAttention, fused GDN and norm/quant kernels are visible. Do not compare
  absolute times to the LM 1023/32 trace. Summary excludes virtual graph
  execute-context ranges to avoid double counting.

Native FP8 reference shapes M=1/2/3/5/256 pass with installed vLLM SM120 kernels.
Further three-layer-shape probes include activation quantization and compare
eager/captured execution with vLLM Marlin. Captured M=1/3/5 native is slower;
M=256 is faster in these shapes. Warm-cache microbenchmarks do not forecast full
model gains or justify blanket replacement. An early probe used wrong scale
strides; required column-major `scales.t()` fixed the harness. Its failure log
is retained. The forced-Marlin helper's generic native-support warning does
not establish hardware capability.

One Marlin M=256,N=5120,K=8704 strict oracle comparison fails for
20/1,310,720 elements, max absolute error 0.01390, normalized RMSE 0.00231.
The failure remains recorded, without relaxed tolerances. This differs from the
passing full-model single-request exact greedy comparisons.

Priorities supported by this evidence: dynamic-batch numerical/state diagnosis,
Qwen GDN chunk Prefill, and TP rank submission imbalance. No new general
Prefill-graph compiler or native W8A8 integration was added.

## Evidence and reproduction

`measurements.json` maps original filenames to complete parsed reports, including
failures and memory phases. Process IDs are removed; timings, tokens and assertions
are preserved. `summary.json` derives medians/ranges/acceptance. `correctness.log`
combines original focused logs with ANSI colors removed. The manifest contains
filenames, sizes, revisions and SHA256; no weights, binaries or full traces are
published. Scripts are archived outside both feature PR diffs.

Scripts use `/root/infini-mtp-5090` with `InfiniLM/`, `InfiniCore/`, `runtime/`,
`models/`, `scripts/`, `tools/`, `logs/`; place `cases.json` in `tools/`.
Initialize pinned submodules, apply the context-free overlay with
`git apply --unidiff-zero core-5090-runtime.patch`, and build with the recorded
environment. Expose matching Python packages/runtime libraries; CUDA headers
must be in `CPATH` and `TRITON_PTXAS_PATH` must point to CUDA `ptxas` in this image.
The following commands run from the indicated source or staging directory.

```bash
# Core; then build/install including _infinicore.
xmake f -y --nv-gpu=y --ccl=y --graph=y --aten=y --cuda=/usr/local/cuda --cuda_arch=sm_120
# LM; then build/install _infinilm.
xmake f -y --cxx11-abi=1
python -m pytest -q test/infinicore/ops/fp8_cast.py
python test/infinicore/ops/paged_attention_prefill.py --nvidia
python test/infiniop/paged_attention.py --nvidia
INFINILM_QWEN_MTP_TEST_MODEL=/models/tiny-qwen-mtp INFINILM_QWEN_MTP_TEST_TP=2 \
  python -m pytest -q test/models/qwen3_5 test/layers/test_pre_transpose.py
# Repeat with TP=1; reuse the previously archived tiny fixture configuration.

# From staging root, idle GPUs, OMP_NUM_THREADS=4.
python scripts/run_measured.py --label infinilm-graph -- \
  python scripts/model_benchmark.py --model models/Qwen3.8-27B-FP8-marlin \
  --mode graph --output logs/infinilm-graph.json
python scripts/run_measured.py --label infinilm-k2 -- \
  python scripts/model_benchmark.py --model models/Qwen3.8-27B-FP8-marlin \
  --mode mtp --candidates 2 --output logs/infinilm-k2.json \
  --reference logs/infinilm-graph.json
# Repeat for eager, K1 eager, K1 mtp-graph and K4 eager.
# Longer comparison: --case math --case english --tokens 128.
python scripts/run_measured.py --label infinilm-graph-memory -- python scripts/validate_graph_memory.py
python scripts/run_measured.py --label infinilm-service-numerics-both -- \
  python scripts/diagnose_service_numerics.py --output logs/infinilm-service-numerics-both.json
# Above reproduction currently exits 1. Retain that failure.
python scripts/run_measured.py --label vllm-graph -- \
  python scripts/model_benchmark.py --framework vllm --model models/Qwen3.8-27B-FP8 \
  --mode graph --output logs/vllm-graph.json
python scripts/run_measured.py --label vllm-k2-diagnostic -- \
  python scripts/model_benchmark.py --framework vllm --model models/Qwen3.8-27B-FP8 \
  --mode mtp-graph --candidates 2 --case zh --tokens 32 --repeats 1 \
  --profile-after --collect-mismatch-profile --reference logs/vllm-graph.json \
  --output logs/vllm-k2-diagnostic.json
# Also exits 1; a diagnostic profile does not make it a pass.
nsys profile --trace=cuda,nvtx --sample=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o logs/infinilm-k2-nsys python scripts/model_benchmark.py \
  --model models/Qwen3.8-27B-FP8-marlin --mode mtp --candidates 2 --case summary \
  --tokens 32 --repeats 1 --profile nsys --reference logs/infinilm-graph.json \
  --output logs/infinilm-k2-profile.json
python scripts/native_fp8_probe.py
python scripts/fp8_cost_probe.py
```

All GPU jobs ended; final device state was 2 MiB and 0% utilization per GPU.
This run does not extend support to random sampling, multimodal/MoE/PP, multiple
MTP layers, other accelerator vendors or partial-prefix reuse.
