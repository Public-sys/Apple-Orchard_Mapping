# Ablation Experiments

This directory is an isolated implementation of the manuscript's eight-group ablation study. The original `Swin-Unet-transLearning/` training files are not modified.

## Experimental matrix

| Exp. | Configuration | DT | CG-PT | PU | SDB | HPL |
|---|---|---:|---:|---:|---:|---:|
| #1 | BS | - | - | - | - | - |
| #2 | DT | ✓ | - | - | - | - |
| #3 | CG-PT | - | ✓ | - | - | - |
| #4 | CG-PT + PU | - | ✓ | ✓ | - | - |
| #5 | CG-PT + PU + SDB | - | ✓ | ✓ | ✓ | - |
| #6 | CG-PT + PU + HPL | - | ✓ | ✓ | - | ✓ |
| #7 | w/o CG-PT + PU + SDB + HPL | - | - | ✓ | ✓ | ✓ |
| #8 | Proposed Full | - | ✓ | ✓ | ✓ | ✓ |

Abbreviations: BS = Baseline (Scratch); DT = Direct Transfer Learning; CG-PT = CycleGAN-based Pre-training; PU = Progressive Unfreezing; SDB = Strategy-Decision Block; HPL = Hierarchical Perceptual Loss.

## Repeated runs

The manuscript protocol is three independent runs with seeds `42`, `1234`, and `2026`. This gives 24 target-domain training runs (8 groups × 3 seeds). For groups using CG-PT, source-domain pre-training is also independently trained for each seed so the complete pipeline is reproducible.

## Run all 24 experiments

Run from the repository root. Replace the example paths with the actual local paths:

```bash
python Ablation_Experiments/scripts/run_ablation.py \
  --target_root D:/data/target \
  --target_list D:/data/target_lists \
  --source_root D:/data/cyclegan_source \
  --source_list D:/data/cyclegan_source_lists \
  --source_ckpt D:/checkpoints/jilin_source_best.pth \
  --cfg Swin-Unet-transLearning/configs/apple_transfer.yaml
```

On Windows CMD, use `^` instead of `\` for line continuation.

`--source_ckpt` is the original source-domain checkpoint required by G2 (DT). `--source_root` and `--source_list` contain the fixed CycleGAN-generated source images/labels used by G3/G4/G5/G6/G8.

## Outputs

Each run is saved independently:

```text
results/
  G1_BS/seed_42/
  G1_BS/seed_1234/
  G1_BS/seed_2026/
  ...
  G8_Full/seed_2026/
```

Each completed run contains `best.pth`, `last.pth`, `history.csv`, `metrics.json`, and `ablation.log`. CG-PT runs additionally contain `source_pretrain/source_best.pth` and the source-pretraining logs.

## Metrics

The unified trainer records PA, Precision, Recall, foreground IoU, and binary mIoU. `evaluate.py` can independently recompute the final metrics from `best.pth`. `mIoU` is explicitly the mean of foreground and background IoU, rather than the foreground IoU used by the legacy trainer's variable name.

After all runs complete:

```bash
python Ablation_Experiments/scripts/summarize_results.py
```

This creates `summary/ablation_mean_std.csv` and `summary/ablation_mean_std.txt`, reporting the sample mean and sample standard deviation over the three independent seeds.

## Important reproducibility rule

Do not manually enter or duplicate the manuscript values such as `83.03 ± 0.12`. The scripts calculate the values from actual independent training runs. If the rerun produces slightly different values, report the actual mean ± SD from the controlled runs.
