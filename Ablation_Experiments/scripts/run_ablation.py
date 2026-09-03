"""Launch all eight ablation settings for three independent random seeds.

Example (Windows PowerShell, from repository root):
python Ablation_Experiments/scripts/run_ablation.py ^
  --target_root D:/data/target ^
  --target_list D:/data/target_lists ^
  --source_root D:/data/cycleGAN_source ^
  --source_list D:/data/cycleGAN_source_lists ^
  --source_ckpt D:/checkpoints/jilin_source_best.pth ^
  --cfg Swin-Unet-transLearning/configs/apple_transfer.yaml

The script does not fabricate results. It starts the actual training processes and
collects their metrics after completion. By default the three seeds are 42, 1234,
and 2026.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT)
SCRIPT_DIR = os.path.join(ROOT, "scripts")
CONFIG_DIR = os.path.join(ROOT, "configs")
TRAINER = os.path.join(SCRIPT_DIR, "train_ablation.py")
SOURCE_TRAINER = os.path.join(REPO_ROOT, "Swin-Unet-transLearning", "train_source.py")

EXPERIMENTS = [
    "G1_BS", "G2_DT", "G3_CGPT", "G4_CGPT_PU",
    "G5_CGPT_PU_SDB", "G6_CGPT_PU_HPL", "G7_PU_SDB_HPL", "G8_Full"
]


def run(cmd):
    print("\n>>>", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Run 8 ablation groups x 3 independent seeds.")
    p.add_argument("--target_root", required=True, help="Target-domain Sentinel-2 image/label root.")
    p.add_argument("--target_list", required=True, help="Target train.txt/val.txt directory.")
    p.add_argument("--source_root", default="", help="CycleGAN-generated source image/label root for CG-PT.")
    p.add_argument("--source_list", default="", help="CycleGAN-generated source train.txt/val.txt directory.")
    p.add_argument("--source_ckpt", default="", help="Original source-domain checkpoint for G2 DT.")
    p.add_argument("--cfg", required=True, help="Swin-Unet config YAML path.")
    p.add_argument("--output_root", default=os.path.join(ROOT, "results"))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1234, 2026])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--base_lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--max_epochs_per_stage", type=int, default=20)
    p.add_argument("--sliding_window", type=int, default=3)
    p.add_argument("--tolerance", type=float, default=0.005)
    p.add_argument("--consecutive", type=int, default=3)
    p.add_argument("--clip_grad", type=float, default=5.0)
    p.add_argument("--stop_on_error", action="store_true")
    args = p.parse_args()

    if len(args.seeds) != 3:
        raise ValueError("For the manuscript protocol, provide exactly three seeds.")
    if not os.path.isfile(args.cfg):
        raise FileNotFoundError(args.cfg)
    if not os.path.isfile(TRAINER):
        raise FileNotFoundError(TRAINER)
    if not args.source_ckpt:
        raise ValueError("--source_ckpt is required because G2 is Direct Transfer Learning.")
    if not (args.source_root and args.source_list):
        raise ValueError("--source_root and --source_list are required because G3/G4/G5/G6/G8 use CG-PT.")

    os.makedirs(args.output_root, exist_ok=True)
    rows = []

    for exp in EXPERIMENTS:
        config_path = os.path.join(CONFIG_DIR, exp + ".json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for seed in args.seeds:
            out_dir = os.path.join(args.output_root, exp, f"seed_{seed}")
            os.makedirs(out_dir, exist_ok=True)

            source_for_run = args.source_ckpt
            if cfg.get("CG_PT", False):
                # Each seed gets an independently trained source-domain checkpoint.
                source_dir = os.path.join(args.output_root, exp, f"seed_{seed}", "source_pretrain")
                source_ckpt = os.path.join(source_dir, "source_best.pth")
                if not os.path.isfile(source_ckpt):
                    run([
                        sys.executable, SOURCE_TRAINER,
                        "--root_path", args.source_root,
                        "--list_dir", args.source_list,
                        "--output_dir", source_dir,
                        "--cfg", args.cfg,
                        "--batch_size", str(args.batch_size),
                        "--epochs", "60",
                        "--img_size", str(args.img_size),
                        "--base_lr", str(args.base_lr),
                        "--min_lr", str(args.min_lr),
                        "--weight_decay", str(args.weight_decay),
                        "--num_workers", str(args.num_workers),
                        "--seed", str(seed)
                    ])
                source_for_run = source_ckpt

            cmd = [
                sys.executable, TRAINER,
                "--config", config_path,
                "--root_path", args.target_root,
                "--list_dir", args.target_list,
                "--output_dir", out_dir,
                "--cfg", args.cfg,
                "--source_ckpt", source_for_run if (cfg.get("DT") or cfg.get("CG_PT")) else "",
                "--img_size", str(args.img_size),
                "--num_classes", "2",
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--base_lr", str(args.base_lr),
                "--min_lr", str(args.min_lr),
                "--weight_decay", str(args.weight_decay),
                "--max_epochs", str(args.max_epochs),
                "--max_epochs_per_stage", str(args.max_epochs_per_stage),
                "--sliding_window", str(args.sliding_window),
                "--tolerance", str(args.tolerance),
                "--consecutive", str(args.consecutive),
                "--perc_weight", str(cfg.get("perc_weight", 0.0)),
                "--clip_grad", str(args.clip_grad),
                "--seed", str(seed)
            ]
            try:
                run(cmd)
                metrics_path = os.path.join(out_dir, "metrics.json")
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                rows.append({"experiment": exp, "seed": seed, **metrics})
            except subprocess.CalledProcessError:
                print(f"FAILED: {exp}, seed={seed}")
                if args.stop_on_error:
                    raise

    all_runs = os.path.join(args.output_root, "all_runs.csv")
    if rows:
        keys = ["experiment", "seed", "mIoU", "F1", "Precision", "Recall", "PA", "IoU_fg", "val_loss"]
        with open(all_runs, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in keys})
    print(f"\nCompleted runs: {len(rows)}. Raw summary: {all_runs}")


if __name__ == "__main__":
    main()
