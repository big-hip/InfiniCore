"""Bounded ordinary B2 timing for the numerical alignment fix (research only)."""

import argparse
import json
import time
from pathlib import Path

import torch
from infinilm.llm.llm import LLM
from infinilm.llm.request import InferenceRequest
from infinilm.llm.sampling_params import SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cases = {c["name"]: c for c in json.loads((args.root / "tools/cases.json").read_text())}
    llm = LLM(
        str(args.root / "models/Qwen3.8-27B-FP8-marlin"),
        enable_mtp=False, max_batch_size=2, num_state_rows=9,
        enable_graph=args.graph, enable_prefix_caching=False,
        device="cuda", dtype="bfloat16", tensor_parallel_size=2,
        num_blocks=80, block_size=64, attn_backend="paged-attn",
        top_k=1, top_p=1.0, weight_load_mode="sync",
    )
    engine = llm.engine

    def run(label, count):
        reqs = [InferenceRequest(
            f"{label}-{name}", prompt_token_ids=cases[name]["prompt_ids"],
            sampling_params=SamplingParams(max_tokens=count, ignore_eos=True, top_k=1),
        ) for name in ("zh", "summary")]
        start = time.perf_counter()
        for req in reqs:
            engine.add_request(req)
        assert engine.step()[0]
        assert all(len(req.generated_token_ids) == 1 for req in reqs)
        first = time.perf_counter()
        while not all(req.is_finished() for req in reqs):
            assert engine.step()[0]
        end = time.perf_counter()
        cache = engine.scheduler.cache_manager
        assert cache.get_total_usable_blocks() == cache.num_blocks
        assert all(b.ref_count == 0 for b in cache.blocks)
        assert not engine.scheduler.mamba_cache_manager.used_block_ids
        outputs = [list(req.generated_token_ids) for req in reqs]
        assert all(len(ids) == count for ids in outputs)
        return dict(output_ids=outputs, ttft_seconds=first-start,
                    decode_seconds=end-first, decode_tps=2*(count-1)/(end-first))

    report = {"graph": args.graph, "batch_size": 2, "output_tokens": 32,
              "prompt_tokens": [63, 1023], "runs": []}
    try:
        run("warmup", 8)
        for repeat in range(3):
            result = run(str(repeat), 32)
            if report["runs"]:
                assert result["output_ids"] == report["runs"][0]["output_ids"]
            report["runs"].append(result)
        report["success"] = True
    finally:
        llm.close()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
