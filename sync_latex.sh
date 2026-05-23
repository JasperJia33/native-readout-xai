#!/usr/bin/env bash
# sync_latex.sh — Sync images and regenerate .tex files for both latex/ and latex_conf/
#
# Usage:
#   bash sync_latex.sh              # sync images + regenerate .tex
#   bash sync_latex.sh --tex-only   # skip image sync, only regenerate .tex
#
# This script:
#   1. Copies all images from outputs/markdown_journal/images/ to latex/en/ and latex/zh/
#   2. Runs md_to_latex.py to regenerate latex/en/main.tex and latex/zh/main.tex
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TEX_ONLY=false
if [[ "$1" == "--tex-only" ]]; then
    TEX_ONLY=true
fi

if [[ "$TEX_ONLY" == false ]]; then
    echo "=== Syncing images ==="
    if [[ -d outputs/markdown_journal/images ]]; then
        mkdir -p latex/en
        cp -u outputs/markdown_journal/images/*.png latex/en/
        echo "  $(ls latex/en/*.png | wc -l) PNG files in latex/en/"
    fi
    if [[ -d outputs/markdown_journal_zh/images ]]; then
        mkdir -p latex/zh
        cp -u outputs/markdown_journal_zh/images/*.png latex/zh/
        echo "  $(ls latex/zh/*.png | wc -l) PNG files in latex/zh/"
    fi
fi

echo "=== Regenerating .tex files ==="
if [[ -f outputs/markdown_journal/report.md ]]; then
    python3 latex/md_to_latex.py --input outputs/markdown_journal/report.md --output latex/en/main.tex
    # Clean auxiliary files before packaging
    rm -f latex/en/main.aux latex/en/main.log latex/en/main.out latex/en/main.pdf
    rm -rf latex/en/.ipynb_checkpoints
    tar -czf latex/en.tar.gz -C latex/en .
    echo "  latex/en.tar.gz ($(du -h latex/en.tar.gz | cut -f1))"
fi
if [[ -f outputs/markdown_journal_zh/report.md ]]; then
    python3 latex/md_to_latex.py --input outputs/markdown_journal_zh/report.md --output latex/zh/main.tex --lang zh
    # Clean auxiliary files before packaging
    rm -f latex/zh/main.aux latex/zh/main.log latex/zh/main.out latex/zh/main.pdf
    rm -rf latex/zh/.ipynb_checkpoints
    tar -czf latex/zh.tar.gz -C latex/zh .
    echo "  latex/zh.tar.gz ($(du -h latex/zh.tar.gz | cut -f1))"
fi

echo "=== Done ==="
