# Current-source C500 validation for InfiniLM #575

The supported MetaX TP1 acceptance checks passed on the restored instance.
No production source or numerical tolerance was changed for this run.

## Revisions and environment

- InfiniLM: `8a591cb1b920dd62fc7f7acf15a8fcdd8458974a`.
- Combined Core: `1f5331b84251818201efee2e827c9c007ce65ca2`, tree
  `aaf6c19603d5d517010d2e2dd35caa02d7d425b2` (#1566 plus #1562).
- One MetaX C500, nominal 64 GiB; MACA 3.7.0.38, driver 3.8.30.
- PyTorch 2.8.0+metax3.7.0.7, Python 3.12.11, Transformers 4.56.2.
- Prepared `state-spaces/mamba2-130m`; TP1. BF16/FP16 activations use FP32
  residuals and SSM state. `INFINIOP_METAX_ALLOW_TF32=0` before startup.
- Fresh Core and LM build directories, with cached Boost 1.90.0 and pybind11
  3.0.4 dependencies. Complete source-file checksums matched the archived
  revisions before and after validation. Loaded extension/runtime hashes are
  recorded in `loaded-libraries.json`.

The first configure attempt tried to download a newer Boost through the
default package repository. It was stopped, and the existing local dependency
repository/cache was registered globally. No project build file was patched.

## Correctness

| Check | Result |
|---|---|
| Mamba BF16 TP1 model/service suite and ordinary Qwen regression | 43 passed |
| Mamba FP32 TP1 continuation, recapture, pool rebuild | 3 passed, 35 intentionally deselected |
| Mamba FP16 TP1 continuation, recapture, pool rebuild | 3 passed, 35 intentionally deselected |
| Scan against independent recurrence, including full state pool | 54 passed |
| Invalid scan descriptors, MetaX GEMM precision, dynamic cat graph | 17 passed |
| `examples/test_infer.py`, MetaX, Decode graph, 32-token limit | Completed successfully |

There were no failed or skipped tests. Python 3.12 emitted SWIG deprecation
warnings; the raw logs retain them. The ordinary synthetic Qwen2 fixture is
FP16 TP1. The BF16 label refers to Mamba. The 43-test suite includes CPU
configuration contracts as well as real accelerator model/service checks;
it is not 43 independent GPU kernels.

Model coverage includes eager Prefill, graph Decode, dynamic request indices,
live-request recapture, eager fallback for uncaptured batches, pool rebuild,
state exhaustion/deferred admission, cancellation, EOS/length release and
state-slot reuse. In particular, ordinary Qwen Decode after recapture passed
at the unchanged `atol=rtol=1e-3` tolerance.

This validates the stated 130M adaptation scope, not every Mamba-2 variant.
There is only one C500 available, so no new C500 TP2 result is claimed. PP,
Ascend/Moore scan backends and hybrid variants remain outside this adaptation.
The archived offline/evaluation/HTTP entrypoint runs were not repeated here;
the current service regression uses `LLMEngine` directly.

## Bounded performance check

BF16 TP1, batch one, 128 input tokens and 128 greedy output tokens, with EOS
stopping disabled. Each process runs a 16-token Decode warmup followed by
three measured requests. Two process pairs ran in eager/graph and then
graph/eager order, giving six measured requests per mode. Prefill remains
eager in both modes. The harness measures wall time through the CPU-returning
generation path, excluding tokenization, loading and graph capture.

| Across six runs per mode | Eager | Decode graph |
|---|---:|---:|
| Median finite-request output rate | 120.45 token/s | 117.64 token/s |
| Median of per-run mean ITL | 8.22 ms | 8.42 ms |
| Output-rate range | 117.97–121.94 token/s | 116.77–118.83 token/s |

The first pair's median rates were 121.74 / 117.10 token/s; the reverse-order
pair's were 119.08 / 118.55 token/s. All 12 measured output hashes matched.
The aggregate graph output rate was about 2.3% lower. This short experiment
does **not** establish a graph speedup on MACA 3.7. The framework already
defaults to eager; keep that default for this measured configuration.

The archived MACA 3.5.3 32,000 MiB slice improvement is a separate historical
result and must not be projected onto this newer full-card environment. The
cause of the performance difference has not been isolated. These runs also
do not measure process peak memory or sustained HTTP-service throughput.

## Reproduction and artifacts

`env.sh`, `build.sh`, `validate.sh`, `validate-core.sh` and `benchmark.py`
record the commands and test harness. `environment.json`, `source-manifest.json`
and `loaded-libraries.json` record source/runtime provenance. Test logs and
JUnit XML retain all assertions and results; only container hostname fields
are removed from the published XML. The result image is a rendering of saved
test output, not a terminal screenshot.

This evidence is published on a separate fork documentation branch. No
experiment scripts, build logs, model weights or fork workflows enter #575.
