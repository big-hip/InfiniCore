"""FP32 oracle, causal/strided/varlen coverage and bounded HD256 timing."""
import json
import os
import time
from pathlib import Path

import infinicore
import torch
from infinicore.ops.paged_attention_prefill import paged_attention_prefill


def native(t):
    return infinicore.strided_from_blob(
        t.data_ptr(), list(t.shape), list(t.stride()),
        dtype=infinicore.utils.to_infinicore_dtype(t.dtype),
        device=infinicore.device(t.device.type, t.device.index or 0))


def cpu(t):
    out = torch.empty(t.shape, dtype=infinicore.utils.to_torch_dtype(t.dtype))
    infinicore.from_torch(out).copy_(t)
    return out.float()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(71)
    results = []
    for dtype in (torch.bfloat16, torch.float16):
        for shape in ([(0, 127)], [(63, 65), (17, 3)], [(0, 1023)]):
            multi = len(shape) > 1
            heads, kv_heads, dim, block = (12 if multi else 24), (2 if multi else 4), 256, 64
            pages_per_seq = max((p+q+block-1)//block for p, q in shape)
            pages = pages_per_seq * len(shape) + 3
            index_dtype = torch.int64 if multi else torch.int32
            table = torch.zeros((len(shape), pages_per_seq + 1), dtype=index_dtype, device='cuda')[:, :pages_per_seq]
            table.copy_(torch.randperm(pages, device='cuda')[:len(shape)*pages_per_seq].reshape(table.shape))
            count = sum(q for p, q in shape)
            q = torch.randn(count, heads*2 if multi else heads, dim, device='cuda', dtype=dtype)
            if multi:
                q = q[:, ::2, :]
            k = torch.randn(pages, kv_heads, block, dim, dtype=dtype, device='cuda')
            v = torch.randn_like(k)
            lengths = torch.tensor([p+c for p, c in shape], dtype=index_dtype, device='cuda')
            offsets = torch.tensor([0]+list(torch.tensor([c for p,c in shape]).cumsum(0).tolist()), dtype=index_dtype, device='cuda')
            alibi = torch.linspace(0, .01, heads, device='cuda') if multi else None
            torch.cuda.synchronize()
            args = list(map(native, (q, k, v, table, lengths, offsets)))
            ialibi = native(alibi) if alibi is not None else None
            def run():
                return paged_attention_prefill(*args, ialibi, scale=dim**-.5)
            references = []
            start = 0
            for seq, (past, queries) in enumerate(shape):
                total = past + queries
                keys = k[table[seq].long()].permute(0,2,1,3).reshape(-1,kv_heads,dim)[:total].repeat_interleave(heads//kv_heads,1).float()
                values = v[table[seq].long()].permute(0,2,1,3).reshape(-1,kv_heads,dim)[:total].repeat_interleave(heads//kv_heads,1).float()
                scores = torch.einsum('qhd,khd->hqk',q[start:start+queries].float(),keys)*dim**-.5
                distance = torch.arange(total,device='cuda')[None,:] - (past+torch.arange(queries,device='cuda')[:,None])
                if alibi is not None:
                    scores += alibi[:,None,None]*distance[None,:,:]
                scores.masked_fill_(distance[None,:,:]>0,-float('inf'))
                references.append(torch.einsum('hqk,khd->qhd',scores.softmax(-1),values))
                start += queries
            expected = torch.cat(references).cpu()
            actual = cpu(run())
            tolerance = .02 if dtype == torch.bfloat16 else .01
            try:
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            except AssertionError as error:
                if os.environ.get('INFINIOP_FLASH_PREFILL_KERNEL') != 'ref':
                    raise
                results.append(dict(dtype=str(dtype),shape=shape,reference_failure=str(error)))
                continue
            # Changing a future value cannot affect the first query.
            past, queries = shape[0]
            page, offset = divmod(past+queries-1, block)
            v[table[0,page], :, offset].add_(10)
            torch.cuda.synchronize()
            after = cpu(run())
            assert torch.equal(actual[0],after[0])
            assert not torch.equal(actual[queries-1],after[queries-1])
            for _ in range(2):
                run()
            infinicore.sync_device()
            begin = time.perf_counter()
            for _ in range(3):
                run()
            infinicore.sync_device()
            result = dict(dtype=str(dtype),shape=shape,strided=multi,alibi=multi,
                          max_abs=(actual-expected).abs().max().item(),
                          causal_pass=True,ms=(time.perf_counter()-begin)*1000/3)
            results.append(result)
            print(json.dumps(result),flush=True)
    suffix=os.environ.get('INFINIOP_FLASH_PREFILL_KERNEL','default')
    Path(__file__).with_name(f'prefill-{suffix}.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
