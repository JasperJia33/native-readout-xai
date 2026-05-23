#!/bin/bash
# Journal-version report regeneration.
# Auto-discovers ALL runs in outputs/, runs quantitative interpretability
# evaluation, rebuilds the journal notebook from build_notebook_journal.py,
# executes it, and exports HTML + Markdown.
#
# Usage:
#   bash rerun_report_journal.sh                    # full pipeline
#   bash rerun_report_journal.sh --skip-interp      # skip interpret_eval (fast)
#   bash rerun_report_journal.sh --skip-train-vis   # skip per-run visualization
#   bash rerun_report_journal.sh --render-only      # only rebuild + execute + export
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG=report_journal.log
SKIP_INTERP=0
SKIP_TRAIN_VIS=0
RENDER_ONLY=0
RUNS=""

for arg in "$@"; do
    case "$arg" in
        --skip-interp)    SKIP_INTERP=1 ;;
        --skip-train-vis) SKIP_TRAIN_VIS=1 ;;
        --render-only)    RENDER_ONLY=1; SKIP_INTERP=1; SKIP_TRAIN_VIS=1 ;;
        *)                RUNS="$RUNS $arg" ;;
    esac
done

echo "=== Journal report generation started at $(date) ===" | tee $LOG

# ---------- Stage 1: per-run visualization ----------
if [ $SKIP_TRAIN_VIS -eq 0 ]; then
    if [ -z "$RUNS" ]; then
        RUNS=$(find outputs -maxdepth 2 -name 'config.yaml' -exec dirname {} \; \
               | grep -v comparison | grep -v report | grep -v interpretability | sort)
    fi
    for RUN_DIR in $RUNS; do
        if [ -f "$RUN_DIR/best_model.pth" ]; then
            echo "--- Visualizing: $RUN_DIR ---" | tee -a $LOG
            python visualize.py --run_dir "$RUN_DIR" 2>&1 | tee -a $LOG
        fi
    done
else
    echo "--- Skipped per-run visualization ---" | tee -a $LOG
fi

# ---------- Stage 2: classification comparison ----------
if [ $RENDER_ONLY -eq 0 ]; then
    echo "--- Running compare.py ---" | tee -a $LOG
    python compare.py 2>&1 | tee -a $LOG
fi

# ---------- Stage 3: quantitative interpretability ----------
if [ $SKIP_INTERP -eq 0 ]; then
    echo "--- Running interpret_eval.py (Del/Ins/Stability) ---" | tee -a $LOG
    python interpret_eval.py 2>&1 | tee -a $LOG
else
    echo "--- Skipped interpret_eval.py ---" | tee -a $LOG
fi

# ---------- Stage 4: rebuild journal notebook from template ----------
echo "--- Rebuilding analysis_journal.ipynb from build_notebook_journal.py ---" | tee -a $LOG
python build_notebook_journal.py 2>&1 | tee -a $LOG

# ---------- Stage 5: execute notebook ----------
mkdir -p outputs/report
echo "--- Executing analysis_journal.ipynb ---" | tee -a $LOG
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 \
    --output analysis_journal_executed.ipynb \
    --output-dir outputs/report \
    analysis_journal.ipynb 2>&1 | tee -a $LOG

# ---------- Stage 6: export HTML ----------
echo "--- Exporting HTML ---" | tee -a $LOG
jupyter nbconvert --to html outputs/report/analysis_journal_executed.ipynb \
    --output analysis_journal_report.html \
    --output-dir outputs/report 2>&1 | tee -a $LOG

# ---------- Stage 7: export Markdown ----------
echo "--- Exporting Markdown ---" | tee -a $LOG
python notebook_to_md.py \
    --input outputs/report/analysis_journal_executed.ipynb \
    --output outputs/markdown_journal 2>&1 | tee -a $LOG

# ---------- Stage 8: generate LaTeX and arXiv tar.gz ----------
echo "--- Syncing LaTeX and packaging arXiv tar.gz ---" | tee -a $LOG
bash sync_latex.sh 2>&1 | tee -a $LOG

echo "=== Journal report generation finished at $(date) ===" | tee -a $LOG
echo ""
echo "Artefacts:"
echo "  Classification:     outputs/comparison/"
echo "  Interpretability:   outputs/interpretability/"
echo "  Executed notebook:  outputs/report/analysis_journal_executed.ipynb"
echo "  HTML report:        outputs/report/analysis_journal_report.html"
echo "  Markdown report:    outputs/markdown_journal/report.md"
echo "  Markdown images:    outputs/markdown_journal/images/"
echo "  LaTeX (EN):         latex/en/main.tex"
echo "  arXiv package (EN): latex/en.tar.gz"
echo "  arXiv package (ZH): latex/zh.tar.gz"
echo "  Log:                $LOG"
