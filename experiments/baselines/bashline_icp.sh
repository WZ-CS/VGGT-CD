#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/path/to/WAT}"
SCENE="${SCENE:-street}"
SEQ_T1="${SEQ_T1:-IMG_9237}"
SEQ_T2="${SEQ_T2:-IMG_9245}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_results_baseline_icp}"
MODEL_PATH="${VGGT_MODEL_PATH:-$PROJECT_ROOT/model.pt}"

python "$SCRIPT_DIR/baseline_icp.py" \
    --data_root "$DATA_ROOT" \
    --scene "$SCENE" \
    --seq_t1 "$SEQ_T1" \
    --seq_t2 "$SEQ_T2" \
    --output_root "$OUTPUT_ROOT" \
    --model_path "$MODEL_PATH"
