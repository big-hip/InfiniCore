"""Summarize device kernels without double-counting CUDA graph annotations."""
import collections
import gzip
import json
from pathlib import Path

root = Path('/root/infini-mtp-5090/logs')
report = {}
for path in sorted((root / 'vllm-profile').glob('rank*.trace.json.gz')):
    events = json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
    stats = collections.defaultdict(lambda: [0, 0])
    excluded = []
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        name = event['name']
        if name.startswith('execute_context_'):
            excluded.append(name)
            continue
        stats[name][0] += float(event.get('dur', 0)) / 1000
        stats[name][1] += 1
    report[path.name.split('.')[0]] = {
        'scope': 'Profiled zh/16-token diagnostic; MTP output check against ordinary failed. Kernel sums exclude virtual execute_context graph ranges. These timings are not production speed.',
        'top_kernels': [{'name': name, 'ms': value[0], 'calls': value[1]}
                        for name, value in sorted(stats.items(), key=lambda item: -item[1][0])[:20]],
        'excluded_virtual_context_count': len(excluded),
    }
(root / 'vllm-k2-profile-summary.json').write_text(json.dumps(report, indent=2) + '\n')
for rank, row in report.items():
    print(rank, row['top_kernels'][:3])
