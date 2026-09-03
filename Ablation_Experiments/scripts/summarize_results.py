"""Aggregate three independent ablation runs as mean +/- sample SD.

Input: results/<group>/seed_<seed>/metrics.json
Output: summary/ablation_mean_std.csv and summary/ablation_mean_std.txt
"""
import argparse
import csv
import json
import math
import os

EXPERIMENTS = [
    "G1_BS", "G2_DT", "G3_CGPT", "G4_CGPT_PU",
    "G5_CGPT_PU_SDB", "G6_CGPT_PU_HPL", "G7_PU_SDB_HPL", "G8_Full"
]
METRICS = ["PA", "Precision", "Recall", "F1", "mIoU"]


def mean_sd(values):
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default="")
    p.add_argument("--output_dir", default="")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1234, 2026])
    args = p.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_root = args.results_root or os.path.join(here, "results")
    output_dir = args.output_dir or os.path.join(here, "summary")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for exp in EXPERIMENTS:
        run_values = {m: [] for m in METRICS}
        for seed in args.seeds:
            path = os.path.join(results_root, exp, f"seed_{seed}", "metrics.json")
            if not os.path.isfile(path):
                print(f"WARNING: missing {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for m in METRICS:
                if m in data:
                    run_values[m].append(float(data[m]))
        row = {"Experiment": exp, "N": min(len(v) for v in run_values.values())}
        for m in METRICS:
            mu, sd = mean_sd(run_values[m])
            row[f"{m}_mean"] = mu
            row[f"{m}_sd"] = sd
            row[f"{m}_mean_sd"] = f"{mu * 100:.2f} ± {sd * 100:.2f}" if not math.isnan(mu) else "NA"
        rows.append(row)

    csv_path = os.path.join(output_dir, "ablation_mean_std.csv")
    fields = ["Experiment", "N"] + [x for m in METRICS for x in (f"{m}_mean", f"{m}_sd", f"{m}_mean_sd")]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    txt_path = os.path.join(output_dir, "ablation_mean_std.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Ablation study: mean +/- sample SD over independent runs\n")
        f.write("Experiment | N | PA | Precision | Recall | F1 | mIoU\n")
        for r in rows:
            f.write(f"{r['Experiment']} | {r['N']} | " + " | ".join(r[f"{m}_mean_sd"] for m in METRICS) + "\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
