"""Record the actual extension and native runtime binaries used for validation."""

import hashlib
import json
import os
from pathlib import Path

import infinicore
import infinilm
from infinicore.lib import _infinicore
from infinilm.lib import _infinilm

root = Path(os.environ["MAMBA575_ROOT"])
paths = {Path(_infinicore.__file__), Path(_infinilm.__file__)}
for line in Path("/proc/self/maps").read_text().splitlines():
    name = line.split()[-1]
    if name.startswith("/") and "/libinfini" in name and ".so" in name:
        paths.add(Path(name))
assert len(paths) >= 6, paths
assert all(path.is_relative_to(root) for path in paths), paths
report = [
    {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in sorted(paths)
]
(root / "evidence/loaded-libraries.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
