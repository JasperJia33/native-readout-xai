#!/usr/bin/env python3
"""
package_release.py — Assemble four clean release bundles from this project.

Each bundle is a self-contained folder under a chosen release directory
that a recipient can copy anywhere and use. No big datasets (LSWMD.pkl,
train/valid/test.pkl, tensorboard event files, log files) are included.

Bundles produced:

  bundle_1_train_and_interpret/
      Minimum to re-run training + interpretability from scratch.
      Recipient brings their own data/LSWMD.pkl.

  bundle_2_rebuild_reports/
      Minimum CSVs + pre-rendered figures to regenerate the notebooks,
      HTML, Markdown, and LaTeX WITHOUT re-running training or
      interpretability.

  bundle_3_figures_only/
      Just the PNGs used by the final report + the source Markdown +
      LaTeX sources. Useful for distributing the deliverable itself.

  bundle_4_model_weights/
      Trained `.pth` weights for all 9 runs + the per-run configs +
      `paper_runs.yaml`, so a recipient can load any model for a one-off
      interpretability experiment without retraining.

Usage:
    python package_release.py --output release
    python package_release.py --output /tmp/my_release --bundles 2,3

The default is to create all four bundles.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Project root = the directory containing this script (latex/ is a sibling).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def copy(src: Path, dst: Path, optional: bool = False) -> bool:
    """Copy a file or directory (recursive). Returns True if copied."""
    src = src.resolve()
    if not src.exists():
        msg = f"  [skip] {src.relative_to(PROJECT_ROOT) if src.is_relative_to(PROJECT_ROOT) else src}"
        if optional:
            print(msg + "  (optional, not present)")
            return False
        print(msg + "  (MISSING — required)")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=_default_ignore)
    else:
        shutil.copy2(src, dst)
    print(f"  [ok]   {src.relative_to(PROJECT_ROOT)} -> {dst.relative_to(dst.parents[len(dst.parents) - 2])}")
    return True


def _default_ignore(directory, contents):
    """Skip cruft when copying a tree."""
    skip = set()
    for name in contents:
        if name in {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}:
            skip.add(name)
        elif name.endswith(".pyc"):
            skip.add(name)
        elif name == "tensorboard":
            skip.add(name)
    return skip


def write_readme(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"  [ok]   wrote {path.name}")


# ---------------------------------------------------------------------------
# Shared assets
# ---------------------------------------------------------------------------
CODE_CORE = [
    "src",
    "utils",
    "configs",
    "configs_mvtec",
    "main.py",
    "main_mvtec.py",
    "compare.py",
    "interpret_eval.py",
    "ablation_eval.py",
    "ablation_final_layer_attn.py",
    "rise_eval.py",
    "main_mvtec.py",
    "interpret_eval_mvtec.py",
    "generate_panel.py",
    "visualize.py",
    "notebook_to_md.py",
    "requirements.txt",
    "run_seeds.sh",
    "rerun_report_journal.sh",
    "rerun_report_journal_zh.sh",
    "sync_latex.sh",
]

NOTEBOOK_BUILDERS = [
    "build_notebook_journal.py",
    "build_notebook_journal_zh.py",
]

LATEX_ASSETS = [
    "latex/md_to_latex.py",
    "latex/Makefile",
    "latex/README.md",
    "latex/en",
    "latex/zh",
]

DOCS = [
    "README.md",
]

PAPER_RUN_DIRS = [
    "outputs/run0_resnet18_cbam",
    "outputs/resnet18_cbam_seed123",
    "outputs/resnet18_cbam_seed456",
    "outputs/run0_densenet121",
    "outputs/densenet121_seed123",
    "outputs/densenet121_seed456",
    "outputs/run0_vit_tiny_v3",
    "outputs/vit_tiny_v3_seed123",
    "outputs/vit_tiny_v3_seed456",
]


# ---------------------------------------------------------------------------
# Bundle 1: full training + interpretability source tree
# ---------------------------------------------------------------------------
def build_bundle_1(dst: Path) -> None:
    print(f"\n=== Bundle 1: train + interpret  ({dst}) ===")
    for item in CODE_CORE + NOTEBOOK_BUILDERS + DOCS:
        copy(PROJECT_ROOT / item, dst / item, optional=("milestone" in item))
    # Data folder placeholder so path exists.
    (dst / "data").mkdir(parents=True, exist_ok=True)
    (dst / "data" / "README.md").write_text(
        "# Data folder\n\n"
        "Place `LSWMD.pkl` here. Download from the WM-811K dataset page:\n"
        "https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map\n\n"
        "After placing the file you can run:\n\n"
        "```bash\n"
        "python main.py --config configs/resnet18_cbam.yaml   # one model\n"
        "bash run_seeds.sh                                     # 6 multi-seed runs\n"
        "python interpret_eval.py                              # faithfulness metrics\n"
        "bash rerun_report_journal.sh                          # full pipeline\n"
        "```\n",
        encoding="utf-8",
    )
    write_readme(
        dst / "README_BUNDLE.md",
        """
# Bundle 1 — Train + Interpret from scratch

This is the full source tree needed to reproduce every experimental
artefact from raw data.

## What's inside
- `src/`, `utils/` — core code (dataset, models, training loop,
  interpretability).
- `configs/` — per-family YAML configs + `paper_runs.yaml` registry.
- `main.py`, `compare.py`, `interpret_eval.py`, `generate_panel.py`,
  `visualize.py`, `notebook_to_md.py` — CLI entry points.
- `run_seeds.sh`, `rerun_report_journal*.sh`, `sync_latex.sh` —
  multi-seed training + full pipeline scripts.
- `build_notebook_journal*.py` — programmatic notebook templates.
- `requirements.txt` — Python dependencies.
- `data/README.md` — where to put `LSWMD.pkl`.

## What's NOT inside
- The WM-811K dataset (`LSWMD.pkl`, ~2 GB). Download separately and put
  it in `data/` before training.
- Trained weights. See `bundle_4_model_weights/` if you want to skip
  training.
- Pre-computed CSVs / figures. See `bundle_2_rebuild_reports/` if you
  want to rebuild reports without running anything.

## Typical workflow
```bash
pip install -r requirements.txt
# put LSWMD.pkl into data/
export CUDA_VISIBLE_DEVICES=0  # select GPU (optional)
python main.py --config configs/resnet18_cbam.yaml
python main.py --config configs/densenet121.yaml
python main.py --config configs/vit_tiny_v3.yaml
bash run_seeds.sh                 # additional seeds 123 and 456
python interpret_eval.py
python rise_eval.py               # RISE model-agnostic control
python ablation_eval.py           # 4 ablation experiments (no retraining)
python ablation_final_layer_attn.py  # final-layer CLS attention ablation
python main_mvtec.py --config configs_mvtec/resnet18_cbam_mvtec.yaml
python main_mvtec.py --config configs_mvtec/densenet121_mvtec.yaml
python main_mvtec.py --config configs_mvtec/vit_tiny_pretrained_mvtec.yaml
python interpret_eval_mvtec.py    # MVTec AD interpretability + RISE
bash rerun_report_journal.sh      # notebook + HTML + markdown + LaTeX + tar.gz
```
""",
    )


# ---------------------------------------------------------------------------
# Bundle 2: rebuild reports without training
# ---------------------------------------------------------------------------
def build_bundle_2(dst: Path) -> None:
    print(f"\n=== Bundle 2: rebuild reports (no training)  ({dst}) ===")
    # Core code needed for compare.py/interpret_eval plot regeneration
    # and for the notebook builder to find the paper runs.
    for item in [
        "src",
        "utils",
        "configs",
        "compare.py",
        "notebook_to_md.py",
        "visualize.py",
        "requirements.txt",
        "rerun_report_journal.sh",
        "rerun_report_journal_zh.sh",
        "sync_latex.sh",
    ] + NOTEBOOK_BUILDERS + DOCS:
        copy(PROJECT_ROOT / item, dst / item, optional=("milestone" in item))

    # Per-run lightweight artefacts only (NO .pth, NO tensorboard).
    for run in PAPER_RUN_DIRS:
        src = PROJECT_ROOT / run
        if not src.exists():
            print(f"  [skip] {run}  (run dir missing)")
            continue
        for fname in ("test_results.json", "training_history.json",
                      "config.yaml", "train.log"):
            s = src / fname
            if s.exists():
                copy(s, dst / run / fname, optional=True)
        # explain_class4 plot (needed by Figures 4a–4c)
        s = src / "plots" / "explain_class4.png"
        if s.exists():
            copy(s, dst / run / "plots" / "explain_class4.png", optional=True)

    # CSVs + summary images needed by the builder.
    interp = PROJECT_ROOT / "outputs/interpretability"
    for fname in ("metrics_per_sample.csv", "summary.csv", "metrics.json",
                  "qualitative_panel.png", "qualitative_panel_raw.png",
                  "deletion_insertion_curves.png",
                  "stability_boxplot.png"):
        copy(interp / fname, dst / "outputs/interpretability" / fname,
             optional=(fname in ("metrics.json", "qualitative_panel_raw.png")))

    # Comparison artefacts (used by notebook).
    comp = PROJECT_ROOT / "outputs/comparison"
    if comp.exists():
        copy(comp, dst / "outputs/comparison")

    # Ablation CSVs (used by notebook builders for supplementary tables).
    ablation = PROJECT_ROOT / "outputs/ablation"
    if ablation.exists():
        copy(ablation, dst / "outputs/ablation")

    # LaTeX converter + Makefile (needed by sync_latex.sh).
    for item in LATEX_ASSETS:
        copy(PROJECT_ROOT / item, dst / item, optional=True)

    # Empty dirs so the pipeline can write into them.
    (dst / "outputs/report").mkdir(parents=True, exist_ok=True)
    (dst / "outputs/markdown_journal").mkdir(parents=True, exist_ok=True)
    (dst / "outputs/markdown_journal_zh").mkdir(parents=True, exist_ok=True)

    write_readme(
        dst / "README_BUNDLE.md",
        """\
# Bundle 2 — Rebuild reports without training

Contains just the CSVs, JSONs, and pre-rendered figures required to
rebuild the journal notebooks, HTML, Markdown, and LaTeX reports.
**No training is needed** — every number in the paper is derivable from
the files in `outputs/interpretability/` and each run's
`test_results.json` / `training_history.json`.

## What's inside
- `src/`, `utils/`, `configs/` — code imported by the notebook
  builders and by the pipeline's rendering stages.
- `build_notebook_journal*.py` — notebook templates.
- `rerun_report_journal.sh`, `rerun_report_journal_zh.sh`,
  `sync_latex.sh` — pipeline scripts.
- `outputs/run*/test_results.json`, `training_history.json`,
  `config.yaml` — per-run classification + curve data.
- `outputs/interpretability/` — per-sample Del/Ins/Stability CSVs +
  the three faithfulness figures + the qualitative heatmap panel.
- `outputs/ablation/` — ablation experiment CSVs (random baseline,
  ViT Grad-CAM, commonly-correct, top-k sensitivity).
- `outputs/comparison/` — classification comparison figures.
- `latex/` — LaTeX converter, Makefile, and arXiv package structure.

## What's NOT inside
- Raw dataset. Not needed for rendering. (Figure 1/Figure 2 in the
  journal notebook require the dataset; see workaround below.)
- Trained weights.
- Tensorboard events / logs.

## Usage
```bash
pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0  # select GPU (optional)

# Render English journal (notebook + HTML + Markdown + LaTeX + tar.gz).
bash rerun_report_journal.sh --render-only

# Render Chinese journal.
bash rerun_report_journal_zh.sh --render-only
```

### Figure 1 / Figure 2 caveat
The journal notebook's Figure 1 (class distribution) and Figure 2
(sample wafer maps) are drawn from the raw WM-811K dataset. If you
don't have `data/LSWMD.pkl`, the notebook will fail at those cells.

**Options:**
1. Download `LSWMD.pkl` into `data/` (~2 GB).
2. Skip those two cells by editing the builder to comment them out,
   or use the pre-rendered `outputs/report/class_distribution.png` and
   `outputs/report/sample_wafers.png` shipped in
   `bundle_3_figures_only/` — copy them into `outputs/report/` here.
""",
    )


# ---------------------------------------------------------------------------
# Bundle 3: final figures + report sources only
# ---------------------------------------------------------------------------
def build_bundle_3(dst: Path) -> None:
    print(f"\n=== Bundle 3: figures + report sources  ({dst}) ===")

    # Final Markdown reports.
    copy(PROJECT_ROOT / "outputs/markdown_journal/report.md",
         dst / "markdown_journal/report.md")
    copy(PROJECT_ROOT / "outputs/markdown_journal/images",
         dst / "markdown_journal/images")
    copy(PROJECT_ROOT / "outputs/markdown_journal_zh/report.md",
         dst / "markdown_journal_zh/report.md",
         optional=True)
    copy(PROJECT_ROOT / "outputs/markdown_journal_zh/images",
         dst / "markdown_journal_zh/images",
         optional=True)

    # HTML reports (pre-built).
    for html in ("analysis_journal_report.html",
                 "analysis_journal_zh_report.html"):
        copy(PROJECT_ROOT / "outputs/report" / html,
             dst / "html" / html, optional=True)

    # Pre-rendered figures used by the journal notebook.
    rep = PROJECT_ROOT / "outputs/report"
    for fname in ("class_distribution.png", "sample_wafers.png",
                  "training_curves_all.png", "f1_radar.png",
                  "classification_summary.csv", "interpretability_summary.csv"):
        copy(rep / fname, dst / "report_figures" / fname, optional=True)

    # Interpretability figures.
    interp = PROJECT_ROOT / "outputs/interpretability"
    for fname in ("qualitative_panel.png", "deletion_insertion_curves.png",
                  "stability_boxplot.png"):
        copy(interp / fname, dst / "report_figures" / fname, optional=True)

    # LaTeX sources + figures + arXiv tar.gz.
    for item in LATEX_ASSETS:
        copy(PROJECT_ROOT / item, dst / item)
    # Include pre-built tar.gz if available.
    for tgz in ("latex/en.tar.gz", "latex/zh.tar.gz"):
        copy(PROJECT_ROOT / tgz, dst / tgz, optional=True)

    # Notebook builders for reference.
    for item in NOTEBOOK_BUILDERS:
        copy(PROJECT_ROOT / item, dst / item, optional=True)

    # Docs.
    for item in DOCS:
        copy(PROJECT_ROOT / item, dst / item, optional=True)

    write_readme(
        dst / "README_BUNDLE.md",
        """\
# Bundle 3 — Figures + report sources (deliverable)

A minimal bundle containing just the final deliverables: Markdown,
HTML, and LaTeX versions of the journal manuscript plus all figures.
Intended for sharing the paper with readers who do not need to
re-run anything.

## What's inside
- `markdown_journal/report.md` + `images/` — English Markdown report.
- `markdown_journal_zh/report.md` + `images/` — Chinese Markdown report.
- `html/analysis_journal_report.html` — rendered English HTML.
- `html/analysis_journal_zh_report.html` — rendered Chinese HTML.
- `report_figures/*.png` — all figures (class distribution, sample
  wafers, F1 radar, training curves, Del/Ins curves, stability
  boxplot, qualitative heatmap panel).
- `latex/en/`, `latex/zh/` — arXiv-ready LaTeX packages (flat).
- `latex/en.tar.gz`, `latex/zh.tar.gz` — arXiv upload archives.
- `build_notebook_journal*.py` — template generators (for reference).
- `README.md` — project documentation.

## Usage
- **Read the paper:** open `html/analysis_journal_report.html` in a
  browser.
- **Build a PDF:** `cd latex && make`
- **Upload to arXiv:** submit `latex/en.tar.gz` or `latex/zh.tar.gz`.
- **Edit the prose:** modify the corresponding `build_notebook_journal*.py`
  template and re-run it (requires Bundle 2 or the full project tree).
""",
    )


# ---------------------------------------------------------------------------
# Bundle 4: model weights + configs
# ---------------------------------------------------------------------------
def build_bundle_4(dst: Path) -> None:
    print(f"\n=== Bundle 4: model weights  ({dst}) ===")

    # Code needed to load the models.
    for item in ["src", "utils", "configs", "configs_mvtec", "requirements.txt",
                 "interpret_eval.py", "interpret_eval_mvtec.py",
                 "rise_eval.py", "ablation_eval.py",
                 "ablation_final_layer_attn.py",
                 "generate_panel.py", "visualize.py"]:
        copy(PROJECT_ROOT / item, dst / item, optional=True)

    # Per-run weights + metadata (NO tensorboard, NO checkpoint.pth).
    total_mb = 0
    for run in PAPER_RUN_DIRS:
        src = PROJECT_ROOT / run
        if not src.exists():
            print(f"  [skip] {run}  (missing)")
            continue
        for fname in ("best_model.pth", "config.yaml", "test_results.json",
                      "training_history.json"):
            s = src / fname
            if s.exists():
                copy(s, dst / run / fname, optional=(fname != "best_model.pth"))
                if fname == "best_model.pth":
                    total_mb += s.stat().st_size / (1024 * 1024)

    # Data folder placeholder (interpretability needs test set).
    (dst / "data").mkdir(parents=True, exist_ok=True)
    (dst / "data" / "README.md").write_text(
        "Place `LSWMD.pkl` here if you want to run interpret_eval.py.\n",
        encoding="utf-8",
    )

    write_readme(
        dst / "README_BUNDLE.md",
        f"""
# Bundle 4 — Trained model weights

9 trained models (3 families × 3 seeds), ready to load for inference
or interpretability experiments.

Total weight file size: ~{total_mb:.0f} MB (across 9 `.pth` files).

## What's inside
Per run directory (`outputs/<run_name>/`):
- `best_model.pth` — trained weights (loaded by `src/models.build_model()`).
- `config.yaml` — exact training configuration.
- `test_results.json` — final test metrics.
- `training_history.json` — per-epoch loss/accuracy.

Plus `src/`, `utils/`, `configs/` to instantiate models and run
`interpret_eval.py` against them.

## What's NOT inside
- `checkpoint.pth` (optimizer state, only needed to RESUME training).
- Tensorboard events.
- Raw data.

## Usage
```python
import torch, yaml
from src.models import build_model

cfg = yaml.safe_load(open('outputs/run0_resnet18_cbam/config.yaml'))
model = build_model(cfg['model'])
state = torch.load('outputs/run0_resnet18_cbam/best_model.pth',
                   map_location='cpu')
model.load_state_dict(state)
model.eval()
```

To run interpretability on these weights (requires data/LSWMD.pkl):
```bash
python interpret_eval.py
```
""",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
BUNDLES = {
    1: ("bundle_1_train_and_interpret", build_bundle_1),
    2: ("bundle_2_rebuild_reports",     build_bundle_2),
    3: ("bundle_3_figures_only",        build_bundle_3),
    4: ("bundle_4_model_weights",       build_bundle_4),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="release",
                    help="Output directory for the bundles (default: release/)")
    ap.add_argument("--bundles", default="1,2,3,4",
                    help="Comma-separated list of bundles to build "
                         "(default: 1,2,3,4)")
    args = ap.parse_args()

    selected = [int(x.strip()) for x in args.bundles.split(",") if x.strip()]
    for b in selected:
        if b not in BUNDLES:
            sys.exit(f"Unknown bundle id: {b}")

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for b in selected:
        name, fn = BUNDLES[b]
        fn(out / name)

    # Top-level README for the release folder.
    write_readme(
        out / "README.md",
        """
# Project release bundles

Four self-contained folders. Pick the one you need — each has its own
`README_BUNDLE.md` with usage instructions.

| Bundle | Purpose | Needs `LSWMD.pkl`? | Needs weights? |
|---|---|---|---|
| `bundle_1_train_and_interpret/` | Re-run everything from raw data | YES (you supply) | No (it trains) |
| `bundle_2_rebuild_reports/`     | Rebuild notebooks/HTML/MD/LaTeX | No* | No |
| `bundle_3_figures_only/`        | Read the paper / build PDF | No | No |
| `bundle_4_model_weights/`       | Load trained models for inference | For interp only | YES (included) |

\\* Bundle 2's Figure 1 / 2 cells need the dataset; see its README for
the workaround using Bundle 3's pre-rendered figures.

None of the bundles include the 2-GB WM-811K dataset.
""",
    )

    # Show size summary.
    print("\n=== Size summary ===")
    for b in selected:
        name = BUNDLES[b][0]
        total = sum(p.stat().st_size for p in (out / name).rglob("*")
                    if p.is_file())
        print(f"  {name:40s}  {total / (1024 * 1024):7.1f} MB")

    print(f"\nDone. Bundles written to: {out}")


if __name__ == "__main__":
    main()
