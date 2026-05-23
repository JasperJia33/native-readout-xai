# Architecture-Aware Explanation Auditing for Industrial Visual Inspection

Multi-model comparison framework for industrial visual inspection, testing the **native-readout hypothesis**: explanation faithfulness depends on structural distance between explainer and model's native decision mechanism. Supports four model families (ResNet18+CBAM, DenseNet121, Swin-Tiny, ViT-Tiny), config-driven experiments, Grad-CAM / Attention-Rollout / RISE explainability, **quantitative faithfulness metrics** (Deletion / Insertion AUC, Stability), **perturbation-baseline sensitivity analysis**, and paper-ready Jupyter reports.

**Paper:** [*Architecture-Aware Explanation Auditing for Industrial Visual Inspection*](https://arxiv.org/abs/2605.14255)

## Setup

```bash
pip install -r requirements.txt
```

Place `LSWMD.pkl` in the `data/` folder.

## Quick Start

```bash
# Select GPU (optional, defaults to all visible devices)
export CUDA_VISIBLE_DEVICES=0

# Train a single model
python main.py --config configs/resnet18_cbam.yaml

# Train with a custom run name and explicit seed
python main.py --config configs/vit_tiny_v3.yaml --run_name vit_exp --seed 123

# Train all 4 families (seed 42, reference runs)
python main.py --config configs/resnet18_cbam.yaml
python main.py --config configs/densenet121.yaml
python main.py --config configs/vit_tiny_v3.yaml
python main.py --config configs/swin_tiny.yaml

# Multi-seed sweep (seeds 123 & 456 × 4 families)
bash run_seeds.sh

# Compare all models (auto-picks every run that has results)
python compare.py

# Quantitative interpretability evaluation (Deletion/Insertion/Stability)
python interpret_eval.py

# RISE model-agnostic control (WM-811K)
python rise_eval.py

# Blur-fill perturbation-baseline sensitivity analysis
python ablation_blur_fill.py

# Ablation experiments (no retraining — uses existing checkpoints)
python ablation_eval.py

# Final-layer CLS attention ablation (rollout-depth control)
python ablation_final_layer_attn.py

# MVTec AD boundary-condition study (exploratory)
python main_mvtec.py --config configs_mvtec/resnet18_cbam_mvtec.yaml
python main_mvtec.py --config configs_mvtec/densenet121_mvtec.yaml
python main_mvtec.py --config configs_mvtec/vit_tiny_pretrained_mvtec.yaml
python main_mvtec.py --config configs_mvtec/swin_tiny_pretrained_mvtec.yaml
python interpret_eval_mvtec.py

# Regenerate qualitative heatmap panel only
python generate_panel.py
```

## Project Structure

```
project/
├── configs/                   # YAML config per model
│   ├── resnet18_cbam.yaml
│   ├── densenet121.yaml
│   ├── vit_tiny_v3.yaml
│   ├── swin_tiny.yaml
│   └── paper_runs.yaml       # Registry: 4 families × 3 seeds
├── configs_mvtec/             # MVTec AD configs
│   ├── resnet18_cbam_mvtec.yaml
│   ├── densenet121_mvtec.yaml
│   ├── vit_tiny_pretrained_mvtec.yaml
│   └── swin_tiny_pretrained_mvtec.yaml
├── data/
│   └── LSWMD.pkl             # Dataset (place manually)
├── src/
│   ├── dataset.py            # Data loading, lot-group split, caching
│   ├── dataset_mvtec.py      # MVTec AD data loading
│   ├── models.py             # Model registry + build_model()
│   ├── models_mvtec.py       # MVTec AD model registry (pretrained)
│   ├── train.py              # Training loop (AMP, cosine LR, focal loss)
│   ├── evaluate.py           # Inference
│   └── interpretability.py   # Heatmap API + Del/Ins AUC + Stability
├── utils/
│   ├── augmentation.py       # Augmentation helpers
│   ├── metrics.py            # FocalLoss, class weights, evaluator
│   └── visualize.py          # GradCAM, AttentionRollout, plotting
├── main.py                   # Train & evaluate one model
├── main_mvtec.py             # Train & evaluate (MVTec AD)
├── compare.py                # Multi-model comparison
├── interpret_eval.py         # Batch interpretability evaluation
├── interpret_eval_mvtec.py   # MVTec AD interpretability + RISE
├── rise_eval.py              # RISE model-agnostic control (WM-811K)
├── ablation_eval.py          # 4 ablation experiments (no retraining)
├── ablation_final_layer_attn.py  # Final-layer CLS attention ablation
├── ablation_blur_fill.py     # Blur-fill perturbation-baseline sensitivity
├── generate_panel.py         # Qualitative heatmap panel (standalone)
├── visualize.py              # Generate plots for one run
├── run_seeds.sh              # Multi-seed training
├── rerun_report.sh           # Full pipeline: compare + interp + HTML
├── package_release.py        # Assemble 4 release bundles
├── requirements.txt
└── README.md
```

## Publication Pipeline

The manuscript is maintained as programmatic notebook builders that generate
Jupyter notebooks → executed notebooks → HTML / Markdown → LaTeX → arXiv tar.gz.

```bash
# Set GPU before running pipelines (optional)
export CUDA_VISIBLE_DEVICES=0

# Full journal pipeline (EN): compare + interp + build + execute + export + LaTeX + tar.gz
bash rerun_report_journal.sh

# Render-only (EN): rebuild notebook from template and export (no recomputation)
bash rerun_report_journal.sh --render-only

# Full journal pipeline (ZH)
bash rerun_report_journal_zh.sh

# Render-only (ZH)
bash rerun_report_journal_zh.sh --render-only

# Sync images, regenerate .tex, and package arXiv tar.gz
bash sync_latex.sh              # sync images + regenerate .tex + tar.gz
bash sync_latex.sh --tex-only   # regenerate .tex + tar.gz only (skip image copy)

# Or manually: generate .tex from Markdown
python3 latex/md_to_latex.py --input outputs/markdown_journal/report.md --output latex/en/main.tex
python3 latex/md_to_latex.py --input outputs/markdown_journal_zh/report.md --output latex/zh/main.tex --lang zh

# Compile PDFs (requires lualatex or xelatex)
cd latex && make          # both EN and ZH
cd latex && make en       # English only
cd latex && make zh       # Chinese only
```

### Publication File Structure

```
project/
├── build_notebook_journal.py          # EN notebook builder (source of truth)
├── build_notebook_journal_zh.py       # ZH notebook builder (source of truth)
├── notebook_to_md.py                  # Executed notebook → Markdown converter
├── rerun_report_journal.sh            # EN end-to-end pipeline script
├── rerun_report_journal_zh.sh         # ZH end-to-end pipeline script
├── sync_latex.sh                      # Sync images + regenerate .tex + package tar.gz
│
├── outputs/
│   ├── report/
│   │   ├── analysis_journal_executed.ipynb    # EN executed notebook
│   │   ├── analysis_journal_report.html       # EN HTML export
│   │   ├── analysis_journal_zh_executed.ipynb # ZH executed notebook
│   │   └── analysis_journal_zh_report.html    # ZH HTML export
│   ├── markdown_journal/              # EN Markdown export
│   │   ├── report.md                  #   manuscript text
│   │   └── images/                    #   extracted figures (fig_00–10)
│   └── markdown_journal_zh/           # ZH Markdown export
│       ├── report.md                  #   manuscript text
│       └── images/                    #   extracted figures
│
└── latex/                             # arXiv-ready LaTeX
    ├── md_to_latex.py                 #   Markdown → LaTeX converter (no pandoc)
    ├── Makefile                       #   Build rules (make en / make zh)
    ├── en/                            #   EN arXiv package (flat)
    │   ├── 00README.XXX               #     arXiv engine directive
    │   ├── main.tex                   #     EN manuscript
    │   └── fig_*.png                  #     figures
    ├── zh/                            #   ZH arXiv package (flat)
    │   ├── 00README.XXX               #     arXiv engine directive
    │   ├── main.tex                   #     ZH manuscript
    │   └── fig_*.png                  #     figures
    ├── en.tar.gz                      #   EN arXiv upload archive
    └── zh.tar.gz                      #   ZH arXiv upload archive
```

### Pipeline Flow

```
build_notebook_journal.py ──→ analysis_journal.ipynb
                                    │  (jupyter nbconvert --execute)
                                    ▼
                          analysis_journal_executed.ipynb
                              │                │
                              ▼                ▼
                    HTML report        notebook_to_md.py
                                              │
                                              ▼
                                outputs/markdown_journal/report.md
                                              │
                                              ▼  (latex/md_to_latex.py)
                                      latex/en/main.tex
                                              │
                                              ▼  (tar -czf)
                                        latex/en.tar.gz
```

## Models

| Model | Params | Description |
|-------|--------|-------------|
| `resnet18_cbam` | ~11M | ResNet-18 + CBAM attention modules |
| `densenet121` | ~7M | DenseNet-121 with dense connectivity |
| `swin_tiny` | ~28M | Swin Transformer (hierarchical, window attention) |
| `vit_tiny` | ~5.5M | Vision Transformer (patch=4, dim=192, depth=12) |

## MVTec AD Extension (Exploratory Boundary-Condition Study)

The project includes a second dataset (MVTec AD) to test boundary conditions of the native-readout hypothesis under pretrained natural-image models.

| Model | Params | Pretrained | Description |
|-------|--------|-----------|-------------|
| `resnet18_cbam_rgb` | ~11M | ImageNet | ResNet-18 + CBAM (RGB) |
| `densenet121_rgb` | ~7M | ImageNet | DenseNet-121 (RGB) |
| `swin_tiny_pretrained_rgb` | ~28M | ImageNet | Swin Transformer (RGB) |
| `vit_tiny_pretrained_rgb` | ~5.5M | ImageNet (DeiT) | Vision Transformer |

All MVTec outputs are stored in `outputs_mvtec/` with configs in `configs_mvtec/`.

## Interpretability Protocol

The project compares explanation **faithfulness** (not just visual appeal) across model families.

### Explanation method per family

| Family | Method | Target layer |
|--------|--------|-------------|
| ResNet18 + CBAM | Grad-CAM | `cbam4` (post-attention) |
| DenseNet121 | Grad-CAM | `features.denseblock4` |
| Swin-Tiny | Grad-CAM | `layers[-1]` (final stage) |
| ViT-Tiny | Attention Rollout | all 12 encoder layers |

### Quantitative metrics

- **Deletion AUC** — progressively zero-out top-ranked pixels; **lower** is more faithful.
- **Insertion AUC** — start from zero image, insert top-ranked pixels; **higher** is more faithful.
- **Stability** — cosine similarity of heatmaps under 5 perturbations (rotation ±15°, translation ±3px, noise σ=0.02); **higher** is more robust.

### Multi-seed protocol

- **WM-811K:** 4 families × 3 seeds = 12 runs. Swin-Tiny uses seeds 7, 123, 456 (seed 42 diverged); others use 42, 123, 456.
- **MVTec AD:** 4 families × 1 seed (42) = 4 runs.

The registry `configs/paper_runs.yaml` defines which runs are included.

## CLI Reference

### main.py — Train & evaluate
```
python main.py --config <path> [--run_name <name>] [--seed <int>]
  --config     Path to YAML config (required)
  --run_name   Custom run folder name (default: {model}_{datetime})
  --seed       Override training seed from config
```

### compare.py — Multi-model comparison
```
python compare.py [--runs <dir1> <dir2> ...]
  --runs       Explicit run directories (default: auto-discover all)
```

### interpret_eval.py — Quantitative interpretability
```
python interpret_eval.py [--paper_runs configs/paper_runs.yaml]
                         [--n_samples 200] [--topk 0.1]
                         [--n_steps 20] [--n_augs 5]
                         [--output outputs/interpretability]
```

### generate_panel.py — Qualitative heatmap panel
```
python generate_panel.py [--paper_runs configs/paper_runs.yaml]
                         [--output outputs/interpretability/qualitative_panel.png]
                         [--reference_family resnet18_cbam]
```

### rise_eval.py — RISE model-agnostic evaluation (WM-811K)
```
python rise_eval.py [--n_samples 200] [--n_masks 4000]
                    [--mask_res 8] [--output outputs/rise]
```

### ablation_blur_fill.py — Perturbation-baseline sensitivity
```
python ablation_blur_fill.py [--n_samples 198] [--sigma 3.0]
                             [--output outputs/ablation]
```
Compares zero-fill vs Gaussian blur-fill Deletion AUC for all four families.

### ablation_final_layer_attn.py — Final-layer CLS attention ablation
```
python ablation_final_layer_attn.py [--seed 42] [--output outputs/ablation]
```
Evaluates final-layer CLS-to-patch attention (single layer, no rollout)
to decompose the ViT faithfulness advantage into readout directness vs
rollout depth. Runs all 3 seeds by default.

### interpret_eval_mvtec.py — MVTec AD interpretability
```
python interpret_eval_mvtec.py [--n_samples 200] [--n_masks 4000]
                               [--mask_res 8] [--output outputs_mvtec/interpretability]
```

### build_notebook_journal.py / build_notebook_journal_zh.py — Notebook builders
```
python build_notebook_journal.py       # generates analysis_journal.ipynb
python build_notebook_journal_zh.py    # generates analysis_journal_zh.ipynb
```

### notebook_to_md.py — Export to Markdown
```
python notebook_to_md.py [--input outputs/report/analysis_journal_executed.ipynb]
                         [--output outputs/markdown_journal]
```

### latex/md_to_latex.py — Markdown to LaTeX
```
python latex/md_to_latex.py --input <markdown_path> --output <tex_path> [--lang zh]
```

## Config Reference

All hyperparameters are in YAML config files. Full schema:

```yaml
model:
  name: resnet18_cbam          # resnet18_cbam | densenet121 | vit_tiny | swin_tiny
  num_classes: 9
  pretrained: true
  dropout: 0.5
  cbam_reduction: 16           # resnet18_cbam only
  cbam_kernel_size: 7          # resnet18_cbam only
  patch_size: 4                # vit_tiny only
  embed_dim: 192               # vit_tiny only
  depth: 12                    # vit_tiny only
  num_heads: 3                 # vit_tiny only
  window_size: 7               # swin_tiny only

data:
  path: data/LSWMD.pkl
  image_size: 64
  test_size: 0.2
  val_size: 0.15
  num_workers: 0
  split_mode: lot_group        # lot_group | stratified
  exclude_none: false
  max_samples_per_class: null
  cache_preprocessed: true

training:
  seed: 42
  batch_size: 32
  batch_size_eval: 64
  epochs: 50
  lr: 0.001
  optimizer: adam               # adam | adamw | sgd
  weight_decay: 0.0
  scheduler:
    type: reduce_on_plateau    # reduce_on_plateau | cosine | step
    factor: 0.5
    patience: 5
  loss: cross_entropy          # cross_entropy | focal
  focal_gamma: 2.0
  early_stop_patience: 10
  early_stop_metric: val_balanced_acc
  grad_clip_norm: 1.0
  mixed_precision: true

augmentation:
  enabled: true
  rotation_prob: 0.7
  flip_prob: 0.5
  noise_prob: 0.3
  noise_std: 0.05
  minority_boost: true
```

## Adding a New Model

1. Add your model class to `src/models.py` with `__init__(self, cfg)` signature
2. Register it: `MODEL_REGISTRY['my_model'] = MyModel`
3. Create `configs/my_model.yaml`
4. Train: `python main.py --config configs/my_model.yaml`

## Troubleshooting

- **`num_workers` shared-memory errors**: Set `data.num_workers: 0`
- **CUDA OOM**: Reduce `training.batch_size` or disable `cache_preprocessed`
- **Slow training on GPU**: Ensure `mixed_precision: true` and `cache_preprocessed: true`
- **Poor minority-class F1**: Try `loss: focal`, `augmentation.minority_boost: true`
