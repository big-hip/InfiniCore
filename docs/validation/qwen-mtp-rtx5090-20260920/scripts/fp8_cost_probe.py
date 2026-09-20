"""Bounded vLLM native-FP8 versus Marlin reference; not InfiniLM timings."""
import json
import statistics
from pathlib import Path

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)


torch.set_num_threads(1)
torch.manual_seed(20260920)
torch.backends.cuda.matmul.allow_tf32 = False


def measure(fn, captured=False):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    group = 32 if captured else 1
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(group):
                fn()
        fn = graph.replay
    times = []
    for _ in range(5):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(100):
            fn()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end) * 10 / group)
    return {'median_us': statistics.median(times), 'range_us': [min(times), max(times)]}


output_path = Path('/root/infini-mtp-5090/logs/fp8-cost-reference.json')
results = json.loads(output_path.read_text())['results'] if output_path.exists() else []
for n, k in [(2560, 5120), (17408, 5120), (5120, 8704)]:
    weight = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * 0.02
    scales = weight.float().view(n // 128, 128, k // 128, 128).abs().amax((1, 3)) / 448
    quantized = (weight.float() / scales.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.float8_e4m3fn)
    # The SM120 blockwise kernel requires column-major weight scales.
    native_scales = scales.t()
    layer = torch.nn.Module()
    layer.output_size_per_partition, layer.input_size_per_partition = n, k
    layer.weight_block_size, layer.orig_dtype = [128, 128], torch.bfloat16
    layer.weight = torch.nn.Parameter(quantized, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    for m in [1, 3, 5, 256]:
        activation = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
        previous = next((row for row in results if row['m'] == m and row['n'] == n and row['k'] == k), None)
        if previous and 'native_captured_including_quant' in previous:
            continue

        def native():
            q, s = per_token_group_quant_fp8(activation, 128, column_major_scales=True)
            return ops.cutlass_scaled_mm(q, quantized.t(), s, native_scales, torch.bfloat16)

        def marlin():
            return apply_fp8_marlin_linear(activation, layer.weight, layer.weight_scale_inv,
                                          layer.workspace, n, k, None)

        q, s = per_token_group_quant_fp8(activation, 128, column_major_scales=True)
        reference = (q.float() * s.repeat_interleave(128, 1)) @ (
            quantized.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)).t()
        out = native()
        torch.testing.assert_close(out.float(), reference.bfloat16().float(), rtol=.02, atol=.01)
        marlin_ref = activation.float() @ (
            quantized.float() * scales.bfloat16().float().repeat_interleave(128, 0).repeat_interleave(128, 1)).t()
        marlin_out = marlin()
        marlin_close_error = None
        try:
            torch.testing.assert_close(marlin_out.float(), marlin_ref.bfloat16().float(), rtol=.02, atol=.01)
        except AssertionError as error:
            marlin_close_error = str(error)
        row = {'m': m, 'n': n, 'k': k, 'native_including_quant': measure(native), 'marlin': measure(marlin),
               'native_normalized_rmse': ((out.float() - reference).square().mean() / reference.square().mean()).sqrt().item(),
               'marlin_normalized_rmse': ((marlin_out.float() - marlin_ref).square().mean() / marlin_ref.square().mean()).sqrt().item(),
               'marlin_strict_close_error': marlin_close_error}
        row['native_captured_including_quant'] = measure(native, captured=True)
        row['marlin_captured'] = measure(marlin, captured=True)
        if previous:
            results.remove(previous)
        results.append(row)
        print(json.dumps(row), flush=True)
        output_path.write_text(json.dumps({
            'scope': 'Both kernels from installed vLLM; GPU-event timing of eager repeated calls; native includes activation quantization. Not an InfiniLM integration or end-to-end speed estimate. Strict-close failures, if any, are retained.',
            'results': results}, indent=2) + '\n')
