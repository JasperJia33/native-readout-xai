#!/bin/bash
# Train 6 additional runs for multi-seed variance estimation.
# The seed=42 runs (run0_*) are already trained; this adds seeds 123 and 456.
# Total 6 new runs, ~2-4 hours depending on model and hardware.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG=run_seeds.log

echo "=== Multi-seed training started at $(date) ===" | tee $LOG

run_one () {
    local CFG=$1
    local SEED=$2
    local NAME=$3
    echo ""                                                           | tee -a $LOG
    echo "--- [$NAME] seed=$SEED config=$CFG ---"                     | tee -a $LOG
    python main.py --config "$CFG" --seed $SEED --run_name "$NAME"    2>&1 | tee -a $LOG
}

# ResNet18+CBAM
run_one configs/resnet18_cbam.yaml 123 resnet18_cbam_seed123
run_one configs/resnet18_cbam.yaml 456 resnet18_cbam_seed456

# DenseNet121
run_one configs/densenet121.yaml 123 densenet121_seed123
run_one configs/densenet121.yaml 456 densenet121_seed456

# ViT-Tiny (use the v3 config, our best ViT variant)
run_one configs/vit_tiny_v3.yaml 123 vit_tiny_v3_seed123
run_one configs/vit_tiny_v3.yaml 456 vit_tiny_v3_seed456

echo ""                                                               | tee -a $LOG
echo "=== Multi-seed training finished at $(date) ==="                | tee -a $LOG
echo "New runs:"                                                      | tee -a $LOG
ls -d outputs/*_seed* outputs/vit_tiny_v3_seed*                       2>&1 | tee -a $LOG
echo ""
echo "Next steps:"
echo "  1. python interpret_eval.py                 # quantitative interpretability"
echo "  2. bash rerun_report.sh                     # regenerate report including new runs"
