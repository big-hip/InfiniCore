"""Real-model TP service lifecycle: packed verification, cancel and re-admit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path('/root/infini-mtp-5090')
HERE=ROOT/'scripts'
def event(name):
    result={'event':name,'unix':time.time()}
    print('BENCH_EVENT '+json.dumps(result),flush=True)
    return result
def resident():
    import subprocess
    return subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip()
import torch
from infinilm.llm.llm import LLM
from infinilm.llm.request import InferenceRequest,RequestStatus
from infinilm.llm.sampling_params import SamplingParams
p=argparse.ArgumentParser();p.add_argument('--tp',type=int,default=2);p.add_argument('--k',type=int,default=2);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(8)
cases={x['name']:x for x in json.loads((ROOT/'tools/cases.json').read_text())}
refs={x['case']:x['output_ids'] for x in json.loads((ROOT/'logs/infinilm-graph.json').read_text())['runs']}
records={'args':vars(args)|{'output':str(args.output)},'events':[event('load_begin')],'runs':[]}
llm=LLM(str(ROOT/'models/Qwen3.8-27B-FP8-marlin'),enable_mtp=True,num_draft_tokens=args.k,max_batch_size=2,num_state_rows=1+2*(args.k+2),enable_graph=False,enable_prefix_caching=False,device='cuda',dtype='bfloat16',tensor_parallel_size=args.tp,num_blocks=80,block_size=64,top_k=1,top_p=1.,weight_load_mode='sync')
engine=llm.engine;runner=engine.model_runner;mtp=runner.speculative_runner
records['events'].append(event('model_ready'))
records['ready_resident_mib']=resident()
calls=[];forward=runner.model_engine.forward_raw
import infinicore
from safetensors import safe_open
candidate_ids = [96746, 114734]
with safe_open(str(ROOT/'models/Qwen3.8-27B-FP8/outside.safetensors'), framework='pt', device='cpu') as handle:
    head = handle.get_slice('lm_head.weight')
    selected_weights = torch.cat([head[token:token+1] for token in candidate_ids]).double()
hidden_records = []
diagnostic_stage = 'initial'
golden_prefix = refs['summary'][:20]


def tracked(**kw):
    if kw.get('target_hidden_states') is None:
        calls.append({'requests':kw['input_offsets'].shape[0]-1,'tokens':kw['input_ids'].shape[-1],'verify':kw.get('sample_all_positions',False)})
    result = forward(**kw)
    if kw.get('target_hidden_states') is None:
        positions = kw['position_ids'].to_numpy()
        if positions.ndim == 2: positions = positions[0]
        matching = [i for i, position in enumerate(positions) if int(position) == 1042]
        if matching:
            ids = kw['input_ids'].to_numpy()[0]
            for index in matching:
                valid_prefix = all(int(ids[i]) == golden_prefix[int(position)-1023]
                                   for i, position in enumerate(positions[:index+1])
                                   if 1023 <= int(position) <= 1042)
                if not valid_prefix: continue
                value = result['hidden_states'].narrow(1, index, 1)
                host = torch.empty(value.shape, dtype=torch.bfloat16)
                infinicore.from_torch(host).copy_(value)
                infinicore.sync_device()
                host = host.reshape(-1).double()
                scores = selected_weights @ host
                record = dict(stage=diagnostic_stage, query_tokens=len(positions),
                              position=1042, candidate_ids=candidate_ids,
                              fp64_dot_from_bf16_hidden_and_weights=scores.tolist(),
                              bf16_rounded_scores=scores.bfloat16().float().tolist())
                hidden_records.append((record, host))
                records.setdefault('numerics', []).append(record)
    return result
runner.model_engine.forward_raw=tracked
def ordinary_with_hidden(**kw):
    return tracked(**kw, sample_all_positions=False, return_logits=False)['output_ids']
runner.model_engine.forward=ordinary_with_hidden

def request(name,case):
    return InferenceRequest(name,prompt_token_ids=cases[case]['prompt_ids'],sampling_params=SamplingParams(max_tokens=32,ignore_eos=True,top_k=1))
def released():
    assert not engine.scheduler.mamba_cache_manager.used_block_ids
    assert all(b.ref_count==0 for b in engine.scheduler.cache_manager.blocks)
def drain(reqs):
    for _ in range(100):
        if all(r.is_finished() for r in reqs):break
        assert engine.step()[0]
    assert all(r.is_finished() for r in reqs)
    released()
def save():args.output.write_text(json.dumps(records,indent=2)+'\n')
try:
    records['events'].append(event('steady_begin'))
    diagnostic_stage = 'packed-ordinary'
    runner.speculative_runner = None
    engine.config.enable_mtp = False
    ordinary = [request('ordinary-packed0', 'zh'), request('ordinary-packed1', 'summary')]
    for req in ordinary: engine.add_request(req)
    drain(ordinary)
    records['packed_ordinary'] = [list(req.generated_token_ids) for req in ordinary]
    records['packed_ordinary_matches_serial'] = [out == refs[name][:32] for out, name in zip(records['packed_ordinary'], ['zh', 'summary'])]
    for name, out in zip(['zh', 'summary'], records['packed_ordinary']): refs[name] = out
    save()
    runner.speculative_runner = mtp
    engine.config.enable_mtp = True
    diagnostic_stage = 'packed-mtp'
    requests=[request('packed0','zh'),request('packed1','summary')];before=(mtp.num_accepted,mtp.num_proposals);start=time.perf_counter()
    for r in requests:engine.add_request(r)
    drain(requests)
    exact=[list(r.generated_token_ids)==refs[c][:32] for r,c in zip(requests,['zh','summary'])]
    records['runs'].append(dict(kind='packed',exact=exact,seconds=time.perf_counter()-start,accepted=mtp.num_accepted-before[0],proposals=mtp.num_proposals-before[1],calls=list(calls),output_ids=[list(r.generated_token_ids) for r in requests]));save()
    assert all(exact),records['runs'][-1]
    assert any(c['requests']==2 and c['verify'] for c in calls)
    lifecycle_outputs = []
    for enabled in [False, True]:
        diagnostic_stage = 'cancel-mtp' if enabled else 'cancel-ordinary'
        runner.speculative_runner = mtp if enabled else None
        engine.config.enable_mtp = enabled
        canceled = request('cancel', 'zh')
        keep = request('keep', 'summary')
        engine.add_request(canceled)
        engine.add_request(keep)
        assert engine.step()[0]
        assert len(keep.generated_token_ids) == 1
        canceled.mark_canceled()
        new = request('new', 'code')
        engine.add_request(new)
        drain([canceled, keep, new])
        assert canceled.status == RequestStatus.CANCELED
        outputs = [list(keep.generated_token_ids), list(new.generated_token_ids)]
        lifecycle_outputs.append(outputs)
        records['runs'].append(dict(kind='cancel_after_prefill_then_arrive', mtp=enabled,
                                   output_ids=outputs, released=True, canceled_status=canceled.status.name))
        save()
    records['lifecycle_mtp_matches_ordinary'] = [a == b for a, b in zip(*lifecycle_outputs)]
    save()
    records['numeric_comparisons'] = []
    for i, (left, lhs) in enumerate(hidden_records):
        for right, rhs in hidden_records[i+1:]:
            records['numeric_comparisons'].append(dict(left=left['stage'], right=right['stage'],
                hidden_equal=torch.equal(lhs, rhs), hidden_max_abs=(lhs-rhs).abs().max().item(),
                hidden_relative_rms=((lhs-rhs).square().mean()/lhs.square().mean()).sqrt().item()))
    save()
    active=request('close-active','zh');engine.add_request(active);engine.step();engine.step();engine.add_request(request('close-waiting','code'))
    records['events'].append(event('steady_end'));records['steady_resident_mib']=resident()
finally:
    llm.close();records['events'].append(event('close'));save()
released();records['lifecycle_cleanup_passed']=True;records['success']=all(records['lifecycle_mtp_matches_ordinary']);save()
assert records['success'], 'Controlled output mismatch is retained; numerical diagnosis and cleanup completed.'
print(json.dumps({k:v for k,v in records.items() if k!='runs'}),flush=True)
