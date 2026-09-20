"""Report per-device GPU kernel time inside measured request steps."""
import argparse
import collections
import json
import sqlite3
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('database', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
c = sqlite3.connect(a.database)
strings = dict(c.execute('select id,value from StringIds'))
ranges = [(start, end, text or strings.get(index, '')) for start, end, text, index in
          c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')]
steps = sorted((s, e) for s, e, name in ranges if name == 'engine/step')


def category(name):
    if 'chunkGatedDeltaRule' in name:
        return 'GDN chunk Prefill'
    if 'recurrent' in name.lower() and 'delta' in name.lower():
        return 'GDN recurrent'
    if 'nccl' in name.lower():
        return 'NCCL (includes rank waiting)'
    if 'marlin' in name.lower():
        return 'FP8 Marlin GEMM'
    if 'attention' in name.lower():
        return 'Attention'
    if any(x in name.lower() for x in ['gemm', 'gemv', 'mma_']):
        return 'Dense GEMM/GEMV'
    return 'Other kernels'


def overlap(s, e, windows):
    return sum(max(0, min(e, end) - max(s, start)) for start, end in windows) / 1e6


def union_ms(items):
    total, stop = 0, 0
    for start, end in sorted(items):
        total += max(0, end - max(stop, start))
        stop = max(stop, end)
    return total / 1e6


report = {'scope': 'Nsight traced request. Kernel sums are per device, not added across devices or to host time. NCCL time includes waiting. GPU interval overlap with host step boundaries is approximate for asynchronous calls.',
          'phases': {}, 'cuda_graph_launch_calls': 0, 'model_host_ranges_ms': {}}
for label, windows in [('prefill_step', steps[:1]), ('subsequent_steps', steps[1:])]:
    times = collections.defaultdict(lambda: collections.defaultdict(float))
    busy = collections.defaultdict(list)
    counts = collections.Counter()
    for start, end, device, name in c.execute('select start,end,deviceId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
        ms = overlap(start, end, windows)
        if ms:
            times[device][category(strings[name])] += ms
            counts[device] += 1
            busy[device].extend((max(start, lo), min(end, hi)) for lo, hi in windows if start < hi and end > lo)
    waits = collections.defaultdict(float)
    for start, end, name in c.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        if 'Synchronize' in strings[name]:
            waits[strings[name]] += overlap(start, end, windows)
    report['phases'][label] = {'step_count': len(windows), 'host_wall_ms': sum(e - s for s, e in windows) / 1e6,
                              'device_kernel_ms': dict(times), 'device_kernel_counts': dict(counts),
                              'device_busy_union_ms': {d: union_ms(v) for d, v in busy.items()},
                              'runtime_wait_ms_summed_across_threads': dict(waits)}
for _, _, name in c.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
    if strings[name].startswith('cudaGraphLaunch'):
        report['cuda_graph_launch_calls'] += 1
for start, end, name in ranges:
    if name.startswith('model/'):
        report['model_host_ranges_ms'].setdefault(name, []).append((end - start) / 1e6)
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
