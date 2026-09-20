#!/usr/bin/env bash
set -euo pipefail
source /data/mamba575-review/env.sh
trap 'mamba575_status=$?; echo "$mamba575_status" > "$MAMBA575_ROOT/evidence/core-validation-exit.txt"' EXIT
cd "$MAMBA575_ROOT/core"
python test/infinicore/ops/mamba2_scan.py --metax \
    > "$MAMBA575_ROOT/evidence/scan-metax.log" 2>&1
python -m pytest -q --maxfail=1 \
    test/infinicore/ops/mamba2_scan.py \
    test/infinicore/ops/test_metax_gemm_precision.py \
    test/infinicore/graph/test_cat.py \
    --junitxml="$MAMBA575_ROOT/evidence/core-regressions.xml" \
    > "$MAMBA575_ROOT/evidence/core-regressions.log" 2>&1
echo CORE_VALIDATION_COMPLETE
