#!/bin/bash
# 中文期刊版本的报告重新生成脚本
# 自动发现 outputs/ 下所有运行目录，运行定量可解释性评估，
# 从 build_notebook_journal_zh.py 重建中文期刊 notebook，
# 执行并导出 HTML + Markdown。
#
# 用法：
#   bash rerun_report_journal_zh.sh                    # 完整流水线
#   bash rerun_report_journal_zh.sh --skip-interp      # 跳过可解释性评估
#   bash rerun_report_journal_zh.sh --skip-train-vis   # 跳过每次运行的可视化
#   bash rerun_report_journal_zh.sh --render-only      # 只重建 + 执行 + 导出
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG=report_journal_zh.log
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

echo "=== 中文期刊报告生成开始 $(date) ===" | tee $LOG

# ---------- Stage 1: per-run visualization ----------
if [ $SKIP_TRAIN_VIS -eq 0 ]; then
    if [ -z "$RUNS" ]; then
        RUNS=$(find outputs -maxdepth 2 -name 'config.yaml' -exec dirname {} \; \
               | grep -v comparison | grep -v report | grep -v interpretability | sort)
    fi
    for RUN_DIR in $RUNS; do
        if [ -f "$RUN_DIR/best_model.pth" ]; then
            echo "--- 可视化运行: $RUN_DIR ---" | tee -a $LOG
            python visualize.py --run_dir "$RUN_DIR" 2>&1 | tee -a $LOG
        fi
    done
else
    echo "--- 跳过每次运行的可视化 ---" | tee -a $LOG
fi

# ---------- Stage 2: classification comparison ----------
if [ $RENDER_ONLY -eq 0 ]; then
    echo "--- 运行 compare.py ---" | tee -a $LOG
    python compare.py 2>&1 | tee -a $LOG
fi

# ---------- Stage 3: quantitative interpretability ----------
if [ $SKIP_INTERP -eq 0 ]; then
    echo "--- 运行 interpret_eval.py (Del/Ins/Stability) ---" | tee -a $LOG
    python interpret_eval.py 2>&1 | tee -a $LOG
else
    echo "--- 跳过 interpret_eval.py ---" | tee -a $LOG
fi

# ---------- Stage 4: rebuild Chinese journal notebook ----------
echo "--- 从 build_notebook_journal_zh.py 重建 analysis_journal_zh.ipynb ---" | tee -a $LOG
python build_notebook_journal_zh.py 2>&1 | tee -a $LOG

# ---------- Stage 5: execute notebook ----------
mkdir -p outputs/report
echo "--- 执行 analysis_journal_zh.ipynb ---" | tee -a $LOG
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 \
    --output analysis_journal_zh_executed.ipynb \
    --output-dir outputs/report \
    analysis_journal_zh.ipynb 2>&1 | tee -a $LOG

# ---------- Stage 6: export HTML ----------
echo "--- 导出 HTML ---" | tee -a $LOG
jupyter nbconvert --to html outputs/report/analysis_journal_zh_executed.ipynb \
    --output analysis_journal_zh_report.html \
    --output-dir outputs/report 2>&1 | tee -a $LOG

# ---------- Stage 7: export Markdown ----------
echo "--- 导出 Markdown ---" | tee -a $LOG
python notebook_to_md.py \
    --input outputs/report/analysis_journal_zh_executed.ipynb \
    --output outputs/markdown_journal_zh 2>&1 | tee -a $LOG

# ---------- Stage 8: 生成 LaTeX 和 arXiv tar.gz ----------
echo "--- 同步 LaTeX 并打包 arXiv tar.gz ---" | tee -a $LOG
bash sync_latex.sh 2>&1 | tee -a $LOG

echo "=== 中文期刊报告生成完成 $(date) ===" | tee -a $LOG
echo ""
echo "产出:"
echo "  分类结果:           outputs/comparison/"
echo "  可解释性结果:       outputs/interpretability/"
echo "  执行后的 notebook:  outputs/report/analysis_journal_zh_executed.ipynb"
echo "  HTML 报告:          outputs/report/analysis_journal_zh_report.html"
echo "  Markdown 报告:      outputs/markdown_journal_zh/report.md"
echo "  Markdown 图片:      outputs/markdown_journal_zh/images/"
echo "  LaTeX (ZH):         latex/zh/main.tex"
echo "  arXiv 包 (EN):      latex/en.tar.gz"
echo "  arXiv 包 (ZH):      latex/zh.tar.gz"
echo "  日志:               $LOG"
