"""Record dedicated-device sampled memory across a model benchmark process tree."""

import argparse
import csv
import datetime
import json
import os
import signal
import subprocess
import time
from pathlib import Path


p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
p.add_argument('--timeout', type=int, default=1200)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
root = Path('/root/infini-mtp-5090/logs')
prefix = root / a.label
command = a.command[1:] if a.command[:1] == ['--'] else a.command
used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                               '--format=csv,noheader,nounits'], text=True).splitlines()
assert len(used) == 2 and all(int(v) < 256 for v in used), f'GPUs are occupied: {used}'
with prefix.with_suffix('.memory.csv').open('w') as mem, prefix.with_suffix('.log').open('w') as log:
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used',
                                '--format=csv,noheader,nounits', '-lms', '100'], stdout=mem)
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'label': a.label, 'pid': child.pid, 'started_unix': time.time()}), flush=True)
    try:
        code = child.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        code = 124
    finally:
        monitor.terminate()
        monitor.wait()
samples = []
for row in csv.reader(prefix.with_suffix('.memory.csv').read_text().splitlines()):
    if len(row) == 3:
        try:
            timestamp = datetime.datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
            samples.append((timestamp, int(row[1]), int(row[2])))
        except ValueError:
            pass
events = [json.loads(line.removeprefix('BENCH_EVENT '))
          for line in prefix.with_suffix('.log').read_text(errors='replace').splitlines()
          if line.startswith('BENCH_EVENT ')]
by_name = {event['event']: event['unix'] for event in events}


def peak(start, end):
    result = {}
    for timestamp, device, used_mib in samples:
        if start - .1 <= timestamp <= end + .1:
            result[device] = max(result.get(device, 0), used_mib)
    return result


phases = {}
for name, start, end in [('load_and_initial_capture', 'load_begin', 'model_ready'),
                         ('warmup', 'warmup_begin', 'steady_begin'),
                         ('steady', 'steady_begin', 'steady_end')]:
    if start in by_name and end in by_name:
        phases[name] = {'seconds': by_name[end] - by_name[start], 'sampled_peak_mib': peak(by_name[start], by_name[end])}
result = {'exit_code': code, 'command': command, 'sample_interval_ms': 100,
          'scope': 'Total device usage on two dedicated GPUs; includes all vLLM worker processes, not only parent PID',
          'initial_mib': [int(v) for v in used], 'overall_peak_mib': peak(0, time.time()), 'phases': phases}
prefix.with_suffix('.memory.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
raise SystemExit(code)
