"""Compare unchanged ordinary model contracts across separately built revisions."""
import argparse
import json
import time
import os
from pathlib import Path
import torch
import infinicore
from infinilm.cache import PagedKVCacheConfig
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--model-type', default='qwen2')
p.add_argument('--graph', action='store_true')
p.add_argument('--steady', action='store_true')
a = p.parse_args()
torch.set_num_threads(1 if a.steady else 8)
if a.steady: os.sched_setaffinity(0, {8,9,10,11})
model_dir = a.out.parent / ('ordinary-' + a.model_type)
model_dir.mkdir(exist_ok=True)
config = dict(model_type=a.model_type, hidden_size=256, intermediate_size=512,
              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
              head_dim=64, vocab_size=512, torch_dtype='float16', rms_norm_eps=1e-5,
              max_position_embeddings=256, rope_theta=10000, hidden_act='silu',
              eos_token_id=0, tie_word_embeddings=True, attention_bias=False)
(model_dir/'config.json').write_text(json.dumps(config))
model = InferEngine(str(model_dir), device=infinicore.device('cuda', 0),
                    cache_config=PagedKVCacheConfig(8, 64, 1),
                    attention_backend='paged-attn', enable_graph_compiling=a.graph)
if not (model_dir/'model.safetensors').exists():
    weights = {}
    generator = torch.Generator().manual_seed(318)
    for name, parameter in sorted(model.state_dict()[0].items()):
        if name == 'lm_head.weight': continue
        tensor = infinicore.Tensor(parameter)
        values = torch.randn(tensor.shape, generator=generator) * 0.03
        if 'norm' in name and name.endswith('weight'): values.fill_(1)
        weights[name] = values.half()
    save_file(weights, model_dir/'model.safetensors')
load_model_state_dict_by_file(model, str(model_dir), dtype=model.dtype)
i32 = lambda x: infinicore.from_list(x, dtype=infinicore.int32)
i64 = lambda x: infinicore.from_list(x, dtype=infinicore.int64)
def run(length):
    prompt = [i % 59 + 1 for i in range(length)]
    ids, logits, times = [], [], []
    for step in range(16):
        past = 0 if step == 0 else length + step - 1
        tokens = prompt if step == 0 else [ids[-1]]
        total = past + len(tokens)
        kw = dict(input_ids=i64([tokens]), position_ids=i64(list(range(past,total))),
                  past_kv_lengths=i32([past]), total_kv_lengths=i32([total]),
                  input_offsets=i32([0,len(tokens)]), cu_seqlens=i32([0,total]),
                  block_tables=i32([list(range((total + 63)//64))]),
                  slot_mapping=i64(list(range(past,total))), sample_all_positions=False, top_k=1)
        start=time.perf_counter()
        out=model.forward_raw(**kw)
        ids.append(int(out['output_ids'].to_numpy()[-1]))
        times.append(time.perf_counter()-start)
        value=torch.empty(out['logits'].shape,dtype=torch.float16)
        infinicore.from_torch(value).copy_(out['logits'])
        infinicore.sync_device()
        logits.append(value.clone())
    return dict(ids=ids, seconds=times),logits
run(4)
if a.steady:
    start=time.perf_counter()
    while time.perf_counter()-start < 1.0:run(63)
report={'model_type':a.model_type,'graph':a.graph,'runs':[]}
outputs={}
for length in [4,63]:
    for repeat in range(5 if a.steady else 3):
        result,logits=run(length)
        report['runs'].append({'length':length,'repeat':repeat,**result})
        outputs[f'{length}-{repeat}']=logits
        print(a.model_type,a.graph,length,repeat,sum(result['seconds']),flush=True)
a.out.write_text(json.dumps(report,indent=2))
torch.save(outputs,a.out.with_suffix('.pt'))
