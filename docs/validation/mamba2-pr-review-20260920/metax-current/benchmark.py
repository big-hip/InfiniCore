"""Short C500 eager/Decode-graph comparison using the existing generate entry."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import infinicore
import torch
from infinilm.cache import PagedKVCacheConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file

here = Path(__file__).resolve().parent
root = Path(os.environ['MAMBA575_ROOT'])
parser = argparse.ArgumentParser()
parser.add_argument('--graph', action='store_true')
args = parser.parse_args()
ids = json.loads((here / 'quality-inputs.json').read_text())['sequences'][0]
load_start = time.perf_counter()
source = str(root / 'models/mamba2-130m')
model = InferEngine(source, device=infinicore.device('cuda', 0),
                    cache_config=PagedKVCacheConfig(64, 256, 4),
                    enable_graph_compiling=args.graph, attention_backend='paged-attn')
load_model_state_dict_by_file(model, source, dtype=model.dtype)
load_seconds = time.perf_counter() - load_start
original = model.forward
stamps = []


def timed_forward(*a, **kw):
    result = original(*a, **kw)
    stamps.append(time.perf_counter())
    return result


model.forward = timed_forward


def run(inputs, output):
    stamps.clear()
    infinicore.sync_device()
    start = time.perf_counter()
    generated = model.generate(inputs, GenerationConfig(max_new_tokens=output,
        top_k=1, top_p=1.0, temperature=1.0, stop_on_eos=False, eos_token_id=[]))
    tokens = torch.stack([torch.as_tensor(t.to_numpy().copy()) for t in generated], dim=1)
    assert len(stamps) == output
    return start, list(stamps), tokens


records = []
for prompt, output, batch in ((128, 128, 1),):
    inputs = infinicore.from_list([ids[:prompt]] * batch, dtype=infinicore.int64)
    run(inputs, 16)
    measurements = []
    for repeat in range(3):
        start, times, tokens = run(inputs, output)
        assert tuple(tokens.shape) == (batch, output)
        measurements.append({
            'repeat': repeat, 'ttft_ms': (times[0] - start) * 1000,
            'mean_itl_ms': (times[-1] - times[0]) / (output - 1) * 1000,
            'total_ms': (times[-1] - start) * 1000,
            'output_tokens_per_second': batch * output / (times[-1] - start),
            'output_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        })
    records.append(dict(prompt=prompt, output=output, batch=batch, measurements=measurements))
    print(json.dumps(records[-1]), flush=True)
free, total = torch.cuda.mem_get_info()
report = dict(device='C500, full device, 65536 MiB', tp=1,
    activation_dtype='bfloat16', state_dtype='float32', decode_graph=args.graph,
    timing='Wall time through CPU availability of every greedy token; excludes tokenization, loading and graph capture.',
    load_and_capture_seconds=load_seconds, device_free_bytes_after_cases=free,
    device_total_bytes=total, memory_note='Device-wide free memory, not a process allocation peak.',
    records=records)
(root / 'evidence' / ('benchmark-graph.json' if args.graph else 'benchmark-eager.json')).write_text(json.dumps(report, indent=2) + '\n')
