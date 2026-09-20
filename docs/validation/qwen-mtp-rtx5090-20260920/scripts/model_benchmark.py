"""Use fixed token inputs and public engine APIs for a bounded TP2 comparison."""

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--framework', choices=['infinilm', 'vllm'], default='infinilm')
    p.add_argument('--model', required=True)
    p.add_argument('--mode', choices=['eager', 'graph', 'mtp', 'mtp-graph'], required=True)
    p.add_argument('--candidates', type=int, default=2)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--case', action='append')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--profile', choices=['nsys', 'vllm'])
    p.add_argument('--profile-after', action='store_true')
    p.add_argument('--collect-mismatch-profile', action='store_true')
    args = p.parse_args()
    if args.collect_mismatch_profile:
        assert args.profile_after and args.framework == 'vllm' and args.repeats == 1
    import torch
    torch.set_num_threads(4)
    root = Path('/root/infini-mtp-5090')
    cases = json.loads((root / 'tools/cases.json').read_text())
    names = args.case or ['zh', 'code', 'summary']
    cases = [c for c in cases if c['name'] in names]
    references = {}
    if args.reference:
        for run in json.loads(args.reference.read_text())['runs']:
            references.setdefault(run['case'], run['output_ids'])
    report = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'pid': os.getpid(), 'runs': [], 'events': [], 'success': False}

    def mark(name):
        entry = {'event': name, 'unix': time.time(), 'pid': os.getpid()}
        report['events'].append(entry)
        print('BENCH_EVENT ' + json.dumps(entry), flush=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    enable_mtp = args.mode.startswith('mtp')
    enable_graph = 'graph' in args.mode
    mark('load_begin')
    if args.framework == 'infinilm':
        from infinilm.llm.llm import LLM
        from infinilm.llm.request import InferenceRequest
        from infinilm.llm.sampling_params import SamplingParams
        llm = LLM(args.model, enable_mtp=enable_mtp, enable_graph=enable_graph,
                  num_draft_tokens=args.candidates, max_batch_size=1,
                  num_state_rows=args.candidates + 3 if enable_mtp else 2,
                  enable_prefix_caching=False, device='cuda', dtype='bfloat16',
                  tensor_parallel_size=2, num_blocks=80, block_size=64,
                  attn_backend='paged-attn', top_k=1, top_p=1.0, weight_load_mode='sync')
        engine = llm.engine
        mtp = engine.model_runner.speculative_runner
        if args.profile == 'nsys':
            from functools import wraps

            def trace_method(fn, label):
                @wraps(fn)
                def wrapped(*values, **options):
                    with torch.cuda.nvtx.range(label):
                        return fn(*values, **options)
                return wrapped

            forward = engine.model_runner.model_engine.forward_raw

            def traced_forward(**options):
                kind = 'draft' if options.get('target_hidden_states') is not None else 'target'
                q = options['input_ids'].shape[-1]
                with torch.cuda.nvtx.range(f'model/{kind}/q={q}'):
                    return forward(**options)

            engine.model_runner.model_engine.forward_raw = traced_forward
            engine.step = trace_method(engine.step, 'engine/step')
            if mtp:
                for name in ('_inputs', '_pack', '_prepare_verify', '_commit', '_draft'):
                    setattr(mtp, name, trace_method(getattr(mtp, name), 'mtp/' + name))

        def generate(case, count, label):
            request = InferenceRequest(label, prompt_token_ids=case['prompt_ids'],
                sampling_params=SamplingParams(max_tokens=count, ignore_eos=True, top_k=1, top_p=1.0))
            before = (mtp.num_proposals, mtp.num_accepted, mtp.num_target_calls) if mtp else (0, 0, 0)
            started, first, steps = time.perf_counter(), None, 0
            engine.add_request(request)
            while not request.is_finished():
                assert engine.step()[0]
                steps += 1
                if first is None and request.generated_token_ids:
                    first = time.perf_counter() - started
            elapsed = time.perf_counter() - started
            cache = engine.scheduler.cache_manager
            assert cache.get_total_usable_blocks() == cache.num_blocks
            assert all(b.ref_count == 0 for b in cache.blocks)
            assert not engine.scheduler.mamba_cache_manager.used_block_ids
            return {'output_ids': list(request.generated_token_ids), 'seconds': elapsed,
                    'ttft_seconds': first, 'steps': steps,
                    'proposals': mtp.num_proposals - before[0] if mtp else 0,
                    'accepted': mtp.num_accepted - before[1] if mtp else 0,
                    'target_calls': mtp.num_target_calls - before[2] if mtp else steps}

        close = llm.close
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, tensor_parallel_size=2, dtype='bfloat16',
                  language_model_only=True, max_model_len=4096, max_num_seqs=1,
                  max_num_batched_tokens=4096, gpu_memory_utilization=0.85,
                  enforce_eager=not enable_graph, enable_prefix_caching=False,
                  speculative_config={'method': 'mtp', 'num_speculative_tokens': args.candidates} if enable_mtp else None,
                  profiler_config={'profiler': 'torch', 'torch_profiler_dir': str(root / 'logs/vllm-profile'),
                                   'torch_profiler_with_stack': False} if args.profile == 'vllm' or args.profile_after else None,
                  seed=20260920, disable_log_stats=False)
        engine = llm.llm_engine

        def generate(case, count, label):
            before_metrics = [dataclasses.asdict(m) for m in llm.get_metrics() if 'spec_decode' in m.name]
            started, first, ids, steps = time.perf_counter(), None, [], 0
            engine.add_request(label, {'prompt_token_ids': case['prompt_ids']},
                               SamplingParams(temperature=0, top_p=1.0, max_tokens=count, ignore_eos=True))
            finished = False
            while not finished:
                outputs = engine.step()
                steps += 1
                for output in outputs:
                    if output.request_id != label:
                        continue
                    if output.outputs:
                        ids = list(output.outputs[0].token_ids)
                        if first is None and ids:
                            first = time.perf_counter() - started
                    finished = output.finished
            elapsed = time.perf_counter() - started
            after_metrics = [dataclasses.asdict(m) for m in llm.get_metrics() if 'spec_decode' in m.name]
            return {'output_ids': ids, 'seconds': elapsed,
                    'ttft_seconds': first, 'steps': steps,
                    'spec_metrics_before': before_metrics, 'spec_metrics_after': after_metrics}

        close = engine.engine_core.shutdown

    try:
        mark('model_ready')
        mark('warmup_begin')
        generate(cases[0], 8, 'warmup')
        if len(cases[-1]['prompt_ids']) > 512:
            generate(cases[-1], 2, 'long-warmup')
        if args.profile == 'nsys':
            torch.cuda.cudart().cudaProfilerStart()
        elif args.profile == 'vllm':
            llm.start_profile()
        mark('steady_begin')
        for repeat in range(args.repeats):
            for case in cases:
                item = generate(case, args.tokens, f"{case['name']}-{repeat}")
                item.update(case=case['name'], repeat=repeat, prompt_tokens=len(case['prompt_ids']))
                assert len(item['output_ids']) == args.tokens
                item['decode_tps'] = (args.tokens - 1) / (item['seconds'] - item['ttft_seconds'])
                if case['name'] in references:
                    item['matches_reference'] = item['output_ids'] == references[case['name']][:args.tokens]
                report['runs'].append(item)
                save()
                print(json.dumps({k: v for k, v in item.items() if k != 'output_ids'}), flush=True)
                if item.get('matches_reference') is False and not args.collect_mismatch_profile:
                    raise AssertionError(f"Output mismatch for {case['name']}; stop before further performance runs")
        mark('steady_end')
        if args.profile == 'nsys':
            torch.cuda.cudart().cudaProfilerStop()
        elif args.profile == 'vllm':
            llm.stop_profile()
        if args.profile_after:
            assert args.framework == 'vllm' and args.profile is None
            mark('separate_profile_begin')
            llm.start_profile()
            profile_run = generate(cases[0], min(16, args.tokens), 'separate-profile')
            llm.stop_profile()
            report['separate_profile_output_ids'] = profile_run['output_ids']
            mark('separate_profile_end')
        if any(item.get('matches_reference') is False for item in report['runs']):
            raise AssertionError('Output mismatch retained; diagnostic profile collected, validation remains failed')
        report['success'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        mark('close')
        save()
        close()


if __name__ == '__main__':
    main()
