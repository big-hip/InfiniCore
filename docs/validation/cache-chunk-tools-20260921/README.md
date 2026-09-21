# Cache/chunk diagnostic archive and test cleanup

This directory belongs to the contributor's validation branch, outside the
InfiniLM feature PRs. It preserves optional diagnostics removed from #573 on
2026-09-21. It does not introduce a new runtime dependency or claim new performance
results.

## Preserved tools

`check_chunk_tp.py`, `check_chunk_output.py` and `graph_counter.cc` are exact copies
from InfiniLM `cb243d9a8599c9b7ce8ce41c9f3f6a395040f4fe`. Together they retain TP/PP
orchestration, KV poisoning and per-rank writes, shared/repeated requests,
cancellation, graph-launch interception, timings and JSON reports. They require
matching built InfiniLM/InfiniCore libraries. KV poisoning is diagnostic
instrumentation, not a performance measurement.

From this directory, with a dense FP16 checkpoint:

```sh
CUDA_VISIBLE_DEVICES=0,1 python check_chunk_tp.py \
  --model /path/to/model --tp 2 --chunk-size 300 --policy slru \
  --output /tmp/chunk-tp2.json
```

For graph-launch counting, build InfiniCore with `--graph=y`, then:

```sh
g++ -std=c++17 -shared -fPIC -I"$INFINI_ROOT/include" \
  graph_counter.cc -ldl -o /tmp/infini-graph-counter.so
LD_PRELOAD=/tmp/infini-graph-counter.so CUDA_VISIBLE_DEVICES=0,1 \
  python check_chunk_tp.py --model /path/to/model --tp 2 \
  --chunk-size 300 --policy slru --graph --output /tmp/chunk-graph.json
```

For PP2 eager, start separate processes using `--tp 1 --pp 2 --stage 0` and
`--tp 1 --pp 2 --stage 1`, one visible GPU per process, matching `--port` values
and distinct output paths. The archived `check_chunk_output.py` adds native
output-suppression and malformed-input checks to the longer lifecycle diagnostic.

## Retained feature tests

#573 now has one short native entrypoint at `test/llm/check_chunk_output.py`.
It compares chunked and ordinary greedy tokens, checks intermediate output
suppression and invalid-input rejection, verifies per-rank cache devices,
requires a real 64-token prefix hit, and checks cancellation/page reclamation.
The ModelRunner output checks moved into `test_chunk_execution.py`. All 83 CPU
tests remain, including LRU/SLRU, admission rollback, remote-KV delayed release,
shared references, configuration forwarding and scheduling progress.

#575 and #584 share identical `test/layers/test_pre_transpose.py` content:
one helper builds Prefill/Decode inputs and copies logits. Packing, tied weights,
live-KV reprocessing/recapture and the original comparison tolerances remain.
The three MTP modules and the Mamba model/service modules were not removed or
rewritten. The #584 README now points to consolidated Core #1566.

No production source changes are part of this cleanup. No new feature PR is
created; the three Core feature PRs are unchanged.

## Bounded verification

| Check | Result |
|---|---|
| #573 CPU suite before and after cleanup | 83 passed in each run |
| #573 clean NVIDIA native build from `cb243d9a` | Passed; existing Core runtime reused |
| Retained short native script, RTX 5090 TP1 eager | Passed |
| Retained short native script, 2 x RTX 5090 TP2 + Decode graphs | Passed |
| Shared ordinary regression on the existing #584 NVIDIA runtime | 1 passed; packing, recapture and Decode logits |
| #575 configuration/remapping/service preflight | 33 passed, 9 real-weight device tests skipped |
| Changed Python formatting and diff whitespace checks | Passed |

The native fixture is seeded Qwen2 FP16: one layer, hidden 128, intermediate 256,
2 attention/KV heads, head dimension 64, vocabulary 128 and tied embeddings.
The chunk test uses 16 pages of 64 tokens, batch 1, SLRU, a 67-token prompt,
4 greedy output tokens and chunk size 17. The repeated prompt must hit 64 tokens.
Its maximum position count is 1024 to satisfy the existing scheduler minimum.

`prepare_dense_fixture.py` and `dense-config.json` reproduce this synthetic
fixture with the #573 runtime (no weights are committed):

```sh
python prepare_dense_fixture.py /tmp/tiny-dense dense-config.json
# Run in the updated InfiniLM checkout with its built libraries selected:
CUDA_VISIBLE_DEVICES=0 python test/llm/check_chunk_output.py \
  --model /tmp/tiny-dense --tp 1
CUDA_VISIBLE_DEVICES=0,1 python test/llm/check_chunk_output.py \
  --model /tmp/tiny-dense --tp 2 --graph
```

The shared ordinary test was executed against the available #584 native runtime,
whose production source remains `78d19f74`; it is not a fresh #575 Mamba GPU run.
The current 5090 Core build lacks the Mamba scan dependency. Historical #575
A6000/C500 model validation still refers to its unchanged production source
`8a591cb1` and is linked in the PR.

The first #575 local CPU attempt hid all CUDA devices. Its CUDA-enabled runtime
failed during import (`cudaGetDeviceCount`, error 100): 19 failures, 14 passed,
10 skipped including the separate ordinary GPU test. The subsequent CPU-only
configuration suite allowed device enumeration, kept the real-model environment
variable unset and passed 33 checks with 9 expected device-test skips. No A6000
model kernels were run for this cleanup.

This short synthetic validation is not a new real-model quality, PP, performance
or graph-launch-counting matrix. Historical benchmark claims remain historical.
See `verification.json` for source/test hashes, runtime fingerprints and captured
result summaries.
