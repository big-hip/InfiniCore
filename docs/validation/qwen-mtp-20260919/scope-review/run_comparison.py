import json,os,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[1]
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1'
variants={'upstream':root/'.worktrees/infinilm-mtp-upstream-check','before':root/'.worktrees/infinilm-mtp-before-check','current':root/'.worktrees/infinilm-pr-mtp'}
def run(variant,script,args,label):
    local=env.copy();local['PYTHONPATH']=str(variants[variant]/'python')+':'+env['PYTHONPATH']
    samples=[]
    with (here/(label+'.log')).open('w') as log:
        process=subprocess.Popen([sys.executable,str(here/script),*args,'--out',str(here/(label+'.json'))],env=local,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
            own=[int(row.split(',')[1]) for row in output.splitlines() if row.split(',')[0].strip()==str(process.pid)]
            if own:samples.append({'unix':time.time(),'mib':sum(own)})
            time.sleep(.5)
    (here/(label+'.memory.json')).write_text(json.dumps({'pid':process.pid,'sample_interval_seconds':.5,'peak_mib':max((s['mib'] for s in samples),default=0),'samples':samples},indent=2))
    print(label,'exit',process.returncode,flush=True)
    if process.returncode:raise RuntimeError(label+' failed; see saved log')
for variant in ['upstream','current']:
    for model in ['qwen2','llama']:
        for graph in [False,True]:
            label=f'{variant}-{model}-'+('graph' if graph else 'eager')
            run(variant,'ordinary_probe.py',['--model-type',model]+(['--graph'] if graph else []),label)
for variant in ['before','current']:
    run(variant,'mtp_perf.py',[],variant+'-mtp-k2')
