"""Validate the installed vLLM block-FP8 kernel as an integration reference."""

import json
from pathlib import Path

import torch
from vllm import _custom_ops as ops


torch.set_num_threads(1)
torch.manual_seed(20260920)
torch.backends.cuda.matmul.allow_tf32 = False
rows = []
for m in [1, 2, 3, 5, 256]:
    n, k = 2560, 5120
    a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    a_scale = a.float().view(m, k // 128, 128).abs().amax(-1).clamp_min(1e-12) / 448
    b_scale = b.float().view(n // 128, 128, k // 128, 128).abs().amax((1, 3)).clamp_min(1e-12) / 448
    qa = (a.float() / a_scale.repeat_interleave(128, 1)).to(torch.float8_e4m3fn)
    qb = (b.float() / b_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.float8_e4m3fn)
    a_scale = a_scale.t().contiguous().t()
    actual = ops.cutlass_scaled_mm(qa, qb.t(), a_scale, b_scale.t(), torch.bfloat16)
    reference = (qa.float() * a_scale.repeat_interleave(128, 1)) @ (
        qb.float() * b_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)).t()
    assert torch.isfinite(actual).all().item()
    torch.testing.assert_close(actual.float(), reference.to(torch.bfloat16).float(), rtol=0.02, atol=0.125)
    err = actual.float() - reference
    row = {'m': m, 'n': n, 'k': k, 'input_dtype': str(qa.dtype), 'weight_dtype': str(qb.dtype),
           'activation_scale_shape': list(a_scale.shape), 'weight_scale_shape': list(b_scale.shape),
           'max_abs_error_vs_fp32': err.abs().max().item(),
           'normalized_rmse': (err.square().mean() / reference.square().mean()).sqrt().item()}
    if m == 3:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            ops.cutlass_scaled_mm(qa, qb.t(), a_scale, b_scale.t(), torch.bfloat16)
            torch.cuda.synchronize()
        trace = '/root/infini-mtp-5090/logs/native-fp8-reference.trace.json'
        profile.export_chrome_trace(trace)
        events = json.loads(Path(trace).read_text()).get('traceEvents', [])
        row['profiled_cuda_kernels'] = sorted({e['name'] for e in events if e.get('cat') == 'kernel'})
    rows.append(row)
    print(json.dumps(row), flush=True)
Path('/root/infini-mtp-5090/logs/native-fp8-reference.json').write_text(json.dumps({
    'scope': 'Installed vLLM block-FP8 kernel reference; not InfiniLM integration or a speed claim',
    'torch': torch.__version__, 'cuda': torch.version.cuda, 'results': rows}, indent=2))
