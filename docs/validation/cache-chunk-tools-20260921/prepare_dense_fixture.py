"""Create the seeded dense checkpoint used for the short native regression."""

import json
import sys
from pathlib import Path

import infinicore
import torch
from infinilm.cache import PagedKVCacheConfig
from infinilm.infer_engine import InferEngine
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
config = json.loads(Path(sys.argv[2]).read_text())
(out / "config.json").write_text(json.dumps(config))
engine = InferEngine(
    str(out),
    device=infinicore.device("cuda", 0),
    cache_config=PagedKVCacheConfig(8, 64, 1),
    attention_backend="paged-attn",
)
generator = torch.Generator().manual_seed(318)
weights = {}
for name, parameter in sorted(engine.state_dict()[0].items()):
    if name == "lm_head.weight":
        continue
    tensor = infinicore.Tensor(parameter)
    value = torch.randn(tensor.shape, generator=generator) * 0.03
    if "norm" in name and name.endswith("weight"):
        value.fill_(1)
    weights[name] = value.to(infinicore.utils.to_torch_dtype(tensor.dtype))
save_file(weights, out / "model.safetensors")
vocab = {"<unk>": 0} | {f"t{i}": i for i in range(1, config["vocab_size"])}
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
    unk_token="<unk>",
    eos_token="<unk>",
)
tokenizer.save_pretrained(out)
print("Dense fixture ready:", out)
