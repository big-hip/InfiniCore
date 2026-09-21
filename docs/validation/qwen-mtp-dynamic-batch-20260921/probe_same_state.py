"""Diagnostic only: compare packed verification with Decode from identical state.

Verification writes scratch rows. Decode then writes the unchanged committed rows;
both consume identical pending tokens and causal past KV. Stop after comparing:
this intentionally consumes the committed states and is not a service test.
"""

import argparse
import json
from pathlib import Path

import infinicore
import torch
from infinilm.llm.llm import LLM
from infinilm.llm.request import InferenceRequest
from infinilm.llm.sampling_params import SamplingParams


class ProbeComplete(Exception):
    pass


def host(value):
    if not isinstance(value, infinicore.Tensor):
        value = infinicore.Tensor(value)
    infinicore.set_device(value.device)
    result = torch.empty(value.shape, dtype=infinicore.utils.to_torch_dtype(value.dtype))
    infinicore.from_torch(result).copy_(value)
    infinicore.sync_device()
    return result


def difference(left, right):
    left, right = left.float(), right.float()
    delta = left - right
    return dict(
        equal=bool(torch.equal(left, right)),
        unequal=int(torch.count_nonzero(delta)),
        max_abs=float(delta.abs().max()),
        relative_rms=float((delta.square().mean() / left.square().mean().clamp_min(1e-30)).sqrt()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tp', type=int, default=2)
    p.add_argument('--k', type=int, default=2)
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--require-exact', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    cases = {c['name']: c['prompt_ids'] for c in json.loads(args.cases.read_text())}
    report = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    llm = LLM(
        args.model, enable_mtp=True, num_draft_tokens=args.k, max_batch_size=2,
        num_state_rows=1 + 2 * (args.k + 2), enable_graph=False,
        enable_prefix_caching=False, device='cuda', dtype='bfloat16',
        tensor_parallel_size=args.tp, num_blocks=80, block_size=64,
        top_k=1, top_p=1.0, weight_load_mode='sync',
    )
    engine = llm.engine
    raw = engine.model_runner.model_engine
    forward = raw.forward_raw
    count = 0

    def tracked(**inputs):
        nonlocal count
        offsets = inputs['input_offsets'].to_numpy().tolist()
        if inputs.get('token_state_indices') is None or len(offsets) != 3:
            return forward(**inputs)
        count += 1
        if count != args.step:
            return forward(**inputs)
        verification = forward(**inputs)
        starts = offsets[:-1]
        pending = host(inputs['input_ids'])[0, starts].tolist()
        positions = inputs['position_ids'].to_numpy()[:, starts].tolist()
        past = inputs['past_kv_lengths'].to_numpy().tolist()
        initial = inputs['mamba_init_state_indices'].to_numpy().tolist()
        destinations = inputs['token_state_indices'].to_numpy().tolist()
        lengths = [n + 1 for n in past]
        slots = inputs['slot_mapping'].to_numpy()[starts].tolist()
        ordinary_inputs = dict(inputs)
        ordinary_inputs.update(
            input_ids=infinicore.from_list([pending], dtype=infinicore.int64),
            position_ids=infinicore.from_list(positions, dtype=infinicore.int64),
            total_kv_lengths=infinicore.from_list(lengths, dtype=infinicore.int32),
            input_offsets=infinicore.from_list([0, 1, 2], dtype=infinicore.int32),
            cu_seqlens=infinicore.from_list([0, lengths[0], sum(lengths)], dtype=infinicore.int32),
            slot_mapping=infinicore.from_list(slots, dtype=infinicore.int64),
            mamba_final_state_indices=inputs['mamba_init_state_indices'],
            token_state_indices=None, sample_all_positions=False, verify_draft=False,
        )
        verify_hidden = host(verification['hidden_states'])[0, starts]
        verify_tokens = host(verification['output_ids']).flatten()[starts].tolist()
        ordinary = forward(**ordinary_inputs)
        decode_hidden = host(ordinary['hidden_states'])[0]
        report['comparison'] = dict(
            past=past, offsets=offsets, initial=initial, destinations=destinations,
            pending_tokens=pending,
            verify_tokens=verify_tokens,
            decode_tokens=host(ordinary['output_ids']).flatten().tolist(),
            hidden=[difference(a, b) for a, b in zip(verify_hidden, decode_hidden)],
        )
        report['states'] = []
        for rank, kinds in enumerate(raw.get_hybrid_states()):
            for kind, layers in enumerate(kinds):
                for layer, state in enumerate(layers):
                    if state is None:
                        continue
                    tensor = infinicore.Tensor(state)
                    for request, start in enumerate(starts):
                        checkpoint = host(tensor.narrow(0, destinations[start], 1))
                        committed = host(tensor.narrow(0, initial[request], 1))
                        report['states'].append(dict(
                            rank=rank, kind=kind, layer=layer, request=request,
                            **difference(checkpoint, committed),
                        ))
        report['diagnostic_completed'] = True
        report['exact'] = all(r['equal'] for r in report['states']) and all(
            r['equal'] for r in report['comparison']['hidden']
        )
        raise ProbeComplete()

    raw.forward_raw = tracked

    def request(name, case):
        return InferenceRequest(name, prompt_token_ids=cases[case], sampling_params=SamplingParams(max_tokens=32, ignore_eos=True, top_k=1))

    try:
        canceled, keep = request('cancel', 'zh'), request('keep', 'summary')
        engine.add_request(canceled)
        engine.add_request(keep)
        assert engine.step()[0]
        canceled.mark_canceled()
        engine.add_request(request('new', 'code'))
        for _ in range(100):
            if not engine.step()[0]:
                break
        raise RuntimeError('Did not reach the requested packed verification step.')
    except ProbeComplete:
        pass
    finally:
        llm.close()
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    for rank in range(args.tp):
        for kind in (0, 1):
            mismatches = [r for r in report['states'] if r['rank'] == rank and r['kind'] == kind and not r['equal']]
            print(json.dumps(dict(rank=rank, kind=kind, first_mismatches=mismatches[:4])), flush=True)
    print(json.dumps(report['comparison']), flush=True)
    if args.require_exact:
        assert report['exact'], 'The same-state comparison must be bitwise exact.'


if __name__ == '__main__':
    main()
