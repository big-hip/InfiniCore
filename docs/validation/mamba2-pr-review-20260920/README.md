# InfiniLM #575 review — 2026-09-20

Reviewed base: `270feb3e` (`InfiniLM-v0.2.9c`). Initial head: `5f89ae3d`.
Reviewed update: `8a591cb1b920dd62fc7f7acf15a8fcdd8458974a`.

## Confirmed finding and correction

The shared paged compiler overwrote physical KV page zero when warming up or
recapturing Decode graphs. A live ordinary Qwen request using that page then
produced different logits. The existing Mamba state guard only protected
convolution and SSM rows. This was an existing Attention recapture limitation,
exposed by reviewing the expanded recapture contract; it was not a defect in the
new Mamba recurrence equations.

Extended the existing packed-linear regression with one ordinary Qwen Decode
after recapture, comparing against the same request without recapture. Before
the fix: 127/128 logits outside the existing `atol=rtol=1e-3` limits, maximum
absolute difference 0.69482421875. After the fix: the same test passed. No test
tolerances changed. Initial reproduction used the pre-existing matching Mamba
runtime; final acceptance uses the consolidated Core integration recorded in
`manifest.json`.

Reused the cache guard to save/restore KV page zero and synchronized before
discarding previous captured graphs. These operations occur during capture,
not every Decode replay. The patch adds no Prefill graph, new graph execution
path, or CPU model-compute fallback. It reuses the same correction already
present in the separate Qwen branch without importing that branch's MTP code.
The focused follow-up changes two existing files (+36/-6); no test module or
feature PR was added.

## Architecture and compatibility review

- Model-specific math and state allocation stay in `csrc/models/mamba2/`;
  processor, remapper and checkpoint preparation follow existing entrypoints.
- Projection sharding uses existing QKV/row-parallel linears, convolution uses
  the existing operator, and Decode uses the existing compiler/state pool.
  TP RMS normalization must reduce across ranks; it cannot use an independent
  per-rank RMSNorm without changing the model.
- The small tied output-head subclass prevents packing from invalidating the
  shared embedding layout. Per-linear packing ownership and deferred loader
  finalization are shared correctness prerequisites, tested on ordinary Qwen.
- The raw-forward parameters are keyword-only, so the added Mamba indices do
  not alter the meaning of existing positional arguments.
- State row zero stays reserved; writable destinations are distinct, graph
  batches are capped to available rows, and uncaptured batches fall back to
  device eager. Index and descriptor contracts rely on the existing scheduler
  and the Core scan checks; this is not a hostile-input security audit.
- Unsupported PP, prefix snapshots, speculative state rollback and remote
  state transfer fail before service worker setup. EOS token zero is retained.
- Three retained test modules cover model/configuration numerics, service
  ownership/lifecycle and shared-linear/recapture compatibility. They cover
  separate failure modes and are not duplicate benchmark scripts.
- Checked changed-code comments/errors, initialization order, RAII usage,
  formatting and third-party dependencies. No vendored upstream Mamba source,
  new dependency license, raw allocation, secret, weight or research artifact
  was introduced into the feature diff. This is not a repository-wide license
  certification.

## Explicit limits

The acceptance checkpoint remains prepared `state-spaces/mamba2-130m`, pure
Mamba-2 with one B/C group and convolution width four. General variants, PP,
Ascend and Moore are outside this adaptation. State-pool sizing remains tied
to logical page count. Prefill remains eager. Low-precision cross-shape logits
are not promised to be bitwise equal. The shared A6000 machine is used for
correctness, not new uncontended latency/throughput measurements.

## CI design

All six open source PRs have fork CI for their current source heads. InfiniLM's
standard push workflow only performs format checks, and its separate Ruff job
checks the formatter script. Neither is a model GPU test. Core's workflow
builds and tests CPU on Ubuntu and Windows.

The separate `ci/mamba2-review-20260920` fork branch adds a CPU build and contract
workflow pinning LM `8a591cb1`, Core runtime `a3ac4df4`, and Core scan/MetaX
`f634435b`. It records the combined Core tree and JUnit output. The feature PR
does not include that fork-specific workflow. GPU results are recorded
separately; a CPU green check never implies Mamba GPU inference passed.

## Current-source acceptance results

- InfiniLM `8a591cb1` built in an empty build directory against consolidated
  Core headers and freshly installed combined runtime. Core was compiled in an
  isolated integration worktree using an existing build cache, with no source
  patch; it is not described as a cache-free Core build.
- A6000 BF16 TP1: **43 passed** (12.70 s).
- A6000 BF16 TP2: **43 passed** (18.78 s).
- A6000 FP32 TP1 continuation, live graph recapture and state-pool rebuild:
  **3 passed**, 35 intentionally deselected (10.35 s). No skips in these runs. The TP/dtype labels refer to Mamba; the ordinary
  synthetic Qwen2 compatibility fixture is FP16 TP1 in both suites.
- Model fixtures exercise eager Prefill and both eager/graph Decode; repeated
  capture preserves active states, and batch five exercises graph fallback.
- Loaded library paths and SHA256 hashes are in `loaded-libraries.json`.
  All InfiniCore runtime libraries resolve to the new review installation.
- Source formatting, whitespace/newline checks and current-head fork format
  and Ruff checks passed. #1562 and #1566 Ubuntu/Windows CPU CI passed.

The first Core build attempt used an unpopulated local CUTLASS directory;
`CUTLASS_ROOT` was then pointed at the already checked-out, exact pinned
submodule revision. LM object compilation overlapped Core compilation; its
initial link waited for the new libraries, then succeeded after installation.
These were build-environment sequencing issues, not source-code fixes.

The former draft blocker of building the current model with its consolidated
dependencies and running it on an accelerator is resolved. Code review is
complete within the stated scope. Fresh MetaX verification was subsequently
completed as recorded below. Upstream required CI remains a merge requirement.

## Fork CPU CI follow-up

The first run, [35508794639](https://github.com/big-hip/InfiniLM/actions/runs/35508794639),
passed both Core and LM builds. The final contracts stage had 14 passed and
19 failed because the new workflow omitted the existing `xxhash` dependency.
The missing dependency was added on the separate CI branch. An optional xmake
cache configuration then failed because this workflow checks projects out to
subdirectories rather than the workspace root; that cache configuration was
removed in `7bfa515c`. Production source and test assertions are unchanged.
[Corrected run 35510643478](https://github.com/big-hip/InfiniLM/actions/runs/35510643478)
completed successfully. Both Core and LM builds passed, followed by **33 passed,
zero failed, zero skipped** CPU contracts in 3.597 seconds. The downloaded
JUnit artifact is `cpu-contracts-passed.xml`; these are configuration, loading
and service-preflight checks, not accelerator inference. The successful run's
Core tree matches the local integration exactly:
`aaf6c19603d5d517010d2e2dd35caa02d7d425b2`.

## Fresh MetaX follow-up

Access was restored and both projects built without source changes on a single
C500 64 GiB, MACA 3.7.0.38, PyTorch 2.8.0+metax3.7.0.7 and Python 3.12.11.
The current head passed 43 BF16 model/service/shared-Qwen checks, plus three
FP32 and three FP16 focused checks. Core passed 54 scan comparisons and 17
additional descriptor/precision/graph tests. There were no failures or skips;
35 tests were intentionally deselected in each focused dtype run. Standard
text inference with Decode graphs also completed. See `metax-current/README.md`
for commands, build provenance, raw results and the exact scope.

The bounded new-runtime benchmark does not show a graph speedup: median
finite-request output rate was 120.45 token/s eager versus 117.64 with Decode
graphs; all 12 measured output hashes matched. The existing eager default
is appropriate for this measured setup. Archived MACA 3.5.3 slice speedups
remain historical results, not a prediction for this new runtime.

This resolves the former missing MetaX verification item within the PR's
claimed scope. Upstream CI approval and Core prerequisites are still required
before merging. No extra production changes or tests were added to #575.
