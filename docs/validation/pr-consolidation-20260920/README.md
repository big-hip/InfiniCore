# PR consolidation review — 2026-09-20

Six active PRs group complete features and their runtime prerequisites. All target
`InfiniLM-v0.2.9c`; no upstream merge, forced push or source-branch deletion.

| Contribution | Retained PR | Change |
|---|---|---|
| Cache policy and chunked execution | InfiniLM #573 | Unchanged |
| Mamba-2 model integration | InfiniLM #575 | Dependency description updated; remains draft |
| Qwen greedy MTP | InfiniLM #584 | Final-scope description and dependency updated |
| MetaX Flash Attention ABI | InfiniCore #1558 | Independent compatibility fix, unchanged |
| Mamba-2 scan and MetaX precision | InfiniCore #1562 | Includes previous #1561, no scan math changes |
| Qwen runtime and recurrent graph correctness | InfiniCore #1566 | Includes previous #1560 |

The three graph commits from #1560 are included as one follow-up commit in #1566;
#1561 is a follow-up commit in #1562. Each destination is a fast-forward of its
original head. The superseded source branches and PR discussions are preserved.
Code/tests are byte-identical to the original PRs, as recorded in
`source-equivalence.json`. These are regrouping changes, not new optimizations.

Final runtime head: `a3ac4df44fc902fe223b441e0255631cc6b73a71`.
Final Mamba operator head: `f634435bb683895e5b084cf1e43f41f28da460ea`.
LM heads unchanged: Mamba `5f89ae3d`, Qwen MTP `86208ae1`.

## Reference submissions

We examined both the descriptions and changed files of merged examples:

- [InfiniLM #565](https://github.com/InfiniTensor/InfiniLM/pull/565): 34 files,
  12 commits, profiling-driven inference improvements and focused regressions.
- [InfiniLM #544](https://github.com/InfiniTensor/InfiniLM/pull/544): 42 files,
  8 commits for one model integration; size alone is not a reason to split.
- [InfiniCore #1546](https://github.com/InfiniTensor/InfiniCore/pull/1546): one PR
  combines paged caching, Decode and Prefill for an end-to-end backend feature.
- [InfiniCore #1544](https://github.com/InfiniTensor/InfiniCore/pull/1544): one PR
  groups inference-blocking runtime defects; unrelated attention support stays separate.
- [InfiniLM #568](https://github.com/InfiniTensor/InfiniLM/pull/568): a small
  independent graph-correctness fix, with narrowly scoped regression coverage.

Use one reviewable outcome per PR, meaningful commits within it, and separate
independent compatibility fixes. Preserve required checklists and negative
results. Do not copy missing template sections or unsupported claims from examples.

## Focused verification

The migrated production/test files match both their original PR and the existing
integrated runtime source. Fresh focused tests used that matching prebuilt
runtime on one A6000 (SM86); this is **not a fresh isolated branch build** or
performance measurement. CUDA Toolkit 12.4, Python 3.11.15, PyTorch 2.9.0+cu128.

- Graph changed-input concatenation, reduction and scalar power: **11 passed**.
- Mamba-2 output and whole-state-pool oracle cases: **54 passed**.
- Mamba-2 invalid descriptor rejection: **9 passed**.
- Both combined Core diffs: repository format check passed using clang-format
  21.1.8 and Ruff 0.15.20; whitespace checks passed.
- MetaX precision was not rerun without a current usable MetaX session. The
  original implementation and process-isolated test are unchanged; earlier
  C500 evidence is linked in #1562.

The first formatting attempt used Black 25.1.0 and found a style difference in
an unchanged Mamba test. The project's current `scripts/format.py` defaults to
Ruff. No source was reformatted to satisfy the wrong tool; the configured
formatter passes. Dependency installation attempts are not validation runs.

Previous fork Windows/Linux Debug/Release matrices for the original four Core
heads all completed successfully (runs 35202907777, 35202907462, 35202910227,
35444866036). They do not substitute for CI on the new combined heads, which is
triggered by these pushes. Upstream fork workflow approval remains maintainer-owned.

## Remaining review boundaries

#1562 can request code review with numerical evidence and explicit pending merge
checks. #575 remains draft for the exact model-head accelerator build and its
remaining documented source/style/license/bounds audit; no unchecked item is
silently declared complete. Requesting review does not mean approval to merge.

Qwen MTP's real RTX 5090 dynamic cancellation/re-admission mismatch remains
unresolved. The single-request gains in the separately archived 5090 report do
not establish full dynamic-batching acceptance. Regrouping does not repair that
failure, add native W8A8, or change Mamba/KV-cache numerical behavior.
