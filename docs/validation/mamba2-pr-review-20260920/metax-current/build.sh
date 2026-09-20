#!/usr/bin/env bash
set -euo pipefail
source /data/mamba575-review/env.sh
trap 'mamba575_status=$?; echo "$mamba575_status" > "$MAMBA575_ROOT/evidence/build-exit.txt"' EXIT
cd "$MAMBA575_ROOT/core"
xmake f -y -m release --metax-gpu=y --use-mc=y --aten=y --ccl=y --graph=y --cpu=y --nv-gpu=n --cudnn=n --ninetoothed=n --use-vendor-ops=n > "$MAMBA575_ROOT/evidence/configure-core.log" 2>&1
xmake build -y -j 8 _infinicore > "$MAMBA575_ROOT/evidence/build-core.log" 2>&1
for mamba575_target in infiniop infinirt infiniccl infinicore_cpp_api _infinicore; do
    xmake install -y "$mamba575_target" >> "$MAMBA575_ROOT/evidence/install-core.log" 2>&1
done
cd "$MAMBA575_ROOT/lm"
xmake f -y -m release --cxx11-abi=1 > "$MAMBA575_ROOT/evidence/configure-lm.log" 2>&1
xmake build -y -j 8 _infinilm > "$MAMBA575_ROOT/evidence/build-lm.log" 2>&1
xmake install -y _infinilm > "$MAMBA575_ROOT/evidence/install-lm.log" 2>&1
python -c 'import infinicore, infinilm, torch; print(infinicore.__file__); print(infinilm.__file__); print(torch.__version__)' > "$MAMBA575_ROOT/evidence/import.log" 2>&1
echo BUILD_COMPLETE
