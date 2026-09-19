"""Short paired 27B check; loading, warmup and snapshots excluded from Decode rate."""
import argparse,json,time,subprocess,os
from pathlib import Path
import torch
from infinilm.llm.llm import LLM
from infinilm.llm.request import InferenceRequest
from infinilm.llm.sampling_params import SamplingParams
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];torch.set_num_threads(8)
cases=json.loads((root/'research/2026-09-18-qwen38-mtp-coverage/cases.json').read_text())
prompt=next(c['prompt_ids'] for c in cases if c['name']=='zh')
llm=LLM(str(root/'research/2026-09-18-qwen38-performance/Qwen3.8-27B-FP8-marlin'),
 enable_mtp=True,num_draft_tokens=2,max_batch_size=1,num_state_rows=5,
 enable_graph=False,enable_prefix_caching=False,device='cuda',dtype='bfloat16',
 num_blocks=40,block_size=64,top_k=1,top_p=1.,weight_load_mode='sync')
e=llm.engine;mtp=e.model_runner.speculative_runner
report={'pid':os.getpid(),'K':2,'TP':1,'runs':[]}
def resident():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
 return [int(row.split(',')[1]) for row in s.splitlines() if row.split(',')[0].strip()==str(os.getpid())]
def run(ids):
 r=InferenceRequest('test',prompt_token_ids=ids,sampling_params=SamplingParams(max_tokens=32,ignore_eos=True,top_k=1))
 ac,pr=mtp.num_accepted,mtp.num_proposals;e.add_request(r);t=time.perf_counter();first=None
 while not r.is_finished():
  assert e.step()[0]
  if first is None:first=time.perf_counter()-t
 elapsed=time.perf_counter()-t
 assert not e.scheduler.mamba_cache_manager.used_block_ids
 assert all(b.ref_count==0 for b in e.scheduler.cache_manager.blocks)
 return dict(prompt_tokens=len(ids),ids=list(r.generated_token_ids),seconds=elapsed,ttft=first,
             decode_tps=31/(elapsed-first),accepted=mtp.num_accepted-ac,proposals=mtp.num_proposals-pr)
try:
 run(prompt)
 report['resident_mib']=resident()
 for ids in [prompt[:4],prompt]:
  for repeat in range(3):
   item=run(ids);report['runs'].append(item);print({k:v for k,v in item.items() if k!='ids'},flush=True)
 a.out.write_text(json.dumps(report,indent=2))
finally:llm.close()
