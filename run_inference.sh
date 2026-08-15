#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/path/to/WAT}"
SCENE="${SCENE:-breville}"
SEQ_T1="${SEQ_T1:-IMG_9184}"
SEQ_T2="${SEQ_T2:-IMG_9185}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results_breville}"
MODEL_PATH="${VGGT_MODEL_PATH:-$SCRIPT_DIR/model.pt}"

python "$SCRIPT_DIR/run_inference.py" \
    --data_root "$DATA_ROOT" \
    --scene "$SCENE" \
    --seq_t1 "$SEQ_T1" \
    --seq_t2 "$SEQ_T2" \
    --output_root "$OUTPUT_ROOT" \
    --model_path "$MODEL_PATH"
