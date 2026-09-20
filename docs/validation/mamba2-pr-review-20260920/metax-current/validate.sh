#!/usr/bin/env bash
set -euo pipefail
source /data/mamba575-review/env.sh
trap 'mamba575_status=$?; echo "$mamba575_status" > "$MAMBA575_ROOT/evidence/validation-exit.txt"' EXIT
cd "$MAMBA575_ROOT/lm"
python -m pytest test/models/mamba2 test/layers/test_pre_transpose.py -q --maxfail=1 \
    --junitxml="$MAMBA575_ROOT/evidence/bf16-tp1.xml" \
    > "$MAMBA575_ROOT/evidence/bf16-tp1.log" 2>&1
for mamba575_dtype in fp32 fp16; do
    INFINILM_MAMBA2_MODEL="$MAMBA575_ROOT/models/mamba2-130m-$mamba575_dtype" \
        python -m pytest test/models/mamba2/test_adaptation.py -q --maxfail=1 \
        -k 'prefill_continuation or cache_rebuild or decode_graph' \
        --junitxml="$MAMBA575_ROOT/evidence/$mamba575_dtype-tp1-focused.xml" \
        > "$MAMBA575_ROOT/evidence/$mamba575_dtype-tp1-focused.log" 2>&1
done
echo VALIDATION_COMPLETE
