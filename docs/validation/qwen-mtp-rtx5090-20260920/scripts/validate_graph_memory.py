"""Real-model draft recapture and cache rebuild with active-state validation."""
import json
import subprocess
import time
from pathlib import Path

import torch
from infinilm.cache import PagedKVCacheConfig
from infinilm.llm.llm import LLM
from infinilm.llm.request import InferenceRequest
from infinilm.llm.sampling_params import SamplingParams

root = Path('/root/infini-mtp-5090')
output = root / 'logs/infinilm-graph-memory.json'
report = {'events': [], 'runs': [], 'success': False}


def mark(name):
    item = {'event': name, 'unix': time.time(), 'resident_mib': subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()}
    report['events'].append(item)
    print('BENCH_EVENT ' + json.dumps(item), flush=True)
    output.write_text(json.dumps(report, indent=2) + '\n')


torch.set_num_threads(4)
case = next(c for c in json.loads((root / 'tools/cases.json').read_text()) if c['name'] == 'zh')
reference = next(c['output_ids'] for c in json.loads((root / 'logs/infinilm-graph.json').read_text())['runs'] if c['case'] == 'zh')
mark('load_begin')
llm = LLM(str(root / 'models/Qwen3.8-27B-FP8-marlin'), enable_mtp=True, enable_graph=True,
          num_draft_tokens=1, max_batch_size=1, num_state_rows=4, enable_prefix_caching=False,
          device='cuda', dtype='bfloat16', tensor_parallel_size=2, num_blocks=80, block_size=64,
          attn_backend='paged-attn', top_k=1, top_p=1., weight_load_mode='sync')
engine, raw = llm.engine, llm.engine.model_runner.model_engine


def check_released():
    cache = engine.scheduler.cache_manager
    assert all(b.ref_count == 0 for b in cache.blocks)
    assert cache.get_total_usable_blocks() == cache.num_blocks
    assert not engine.scheduler.mamba_cache_manager.used_block_ids


try:
    mark('model_ready')
    mark('steady_begin')
    for repeat in range(3):
        req = InferenceRequest(f'recapture-{repeat}', prompt_token_ids=case['prompt_ids'],
                               sampling_params=SamplingParams(max_tokens=32, ignore_eos=True, top_k=1))
        engine.add_request(req)
        assert engine.step()[0] and engine.step()[0]
        mark(f'recapture_{repeat}_begin')
        raw.compile()
        mark(f'recapture_{repeat}_end')
        while not req.is_finished():
            assert engine.step()[0]
        exact = list(req.generated_token_ids) == reference[:32]
        report['runs'].append({'kind': 'active_recapture', 'repeat': repeat, 'exact': exact})
        assert exact
        check_released()
    mark('rebuild_begin')
    generation = raw.cache_generation
    raw.reset_cache(PagedKVCacheConfig(80, 64, 1, 4))
    assert raw.cache_generation > generation
    mark('rebuild_end')
    req = InferenceRequest('after-rebuild', prompt_token_ids=case['prompt_ids'],
                           sampling_params=SamplingParams(max_tokens=32, ignore_eos=True, top_k=1))
    engine.add_request(req)
    while not req.is_finished():
        assert engine.step()[0]
    exact = list(req.generated_token_ids) == reference[:32]
    report['runs'].append({'kind': 'after_rebuild', 'exact': exact})
    assert exact
    check_released()
    mark('steady_end')
    report['success'] = True
finally:
    llm.close()
    mark('close')
