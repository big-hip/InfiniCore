"""Record reproducible runtime details without connection or host identifiers."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import torch

root = Path(os.environ["MAMBA575_ROOT"])
properties = torch.cuda.get_device_properties(0)
report = {
    "source": json.loads((root / "source-manifest.json").read_text()),
    "python": platform.python_version(),
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "numpy", "pytest", "safetensors", "janus")
    },
    "device": {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "count": torch.cuda.device_count(),
        "scope": "Single C500 64 GiB, TP1; no TP2 claim.",
    },
    "sdk": "MACA 3.7.0.38",
    "driver": "3.8.30",
    "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
    "strict_fp32": os.environ.get("INFINIOP_METAX_ALLOW_TF32") == "0",
    "model_sha256": hashlib.sha256(
        (root / "models/mamba2-130m/model.safetensors").read_bytes()
    ).hexdigest(),
    "build_dependencies": {"boost": "1.90.0", "pybind11": "3.0.4"},
    "build": "Fresh Core and LM build directories; cached Boost/pybind11 packages. No project source changes.",
}
verified = subprocess.run(
    ["sha256sum", "-c", "--quiet", "source-files.sha256"],
    cwd=root,
    capture_output=True,
    text=True,
)
report["source_files_match_archive"] = verified.returncode == 0
assert verified.returncode == 0, verified.stdout + verified.stderr
(root / "evidence/environment.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
