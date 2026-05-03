#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fusion_das_benchmark import (
    load_split_entries,
    set_seed,
    train_cnn,
    warm_branch_cache,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs_ablation_inception_patchx_v1"
DATA_ROOT = Path("/Volumes/Data/das_data")
BRANCH_A = [0, 2, 4, 6, 8, 10]
BRANCH_B = [1, 3, 5, 7, 9, 11]
SEED = 42


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_entries = load_split_entries(DATA_ROOT, "train")
    test_entries = load_split_entries(DATA_ROOT, "test")
    warm_branch_cache(
        DATA_ROOT,
        train_entries + test_entries,
        [BRANCH_A, BRANCH_B],
        downsample=16,
    )

    variants = [
        ("full", set()),
        ("no_inception", {"no_inception"}),
        ("no_patch", {"no_patch"}),
        ("no_mix_gate", {"no_mix_gate"}),
        ("no_cross_attention", {"no_cross_attention"}),
        ("no_aux_heads", {"no_aux_heads"}),
    ]

    rows: list[dict[str, object]] = []
    for name, tags in variants:
        run_dir = OUTPUT_DIR / name
        report_path = run_dir / f"fusion_inception_patchx_{name}_report.json"
        if report_path.exists():
            result = json.loads(report_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "variant": name,
                    "ablation_tags": sorted(tags),
                    "accuracy": result["accuracy"],
                    "f1_macro": result["f1_macro"],
                    "nar": result["nar"],
                    "fnr": result["fnr"],
                    "latency_ms_per_sample": result["latency_ms_per_sample"],
                }
            )
            continue
        result = train_cnn(
            DATA_ROOT,
            train_entries,
            test_entries,
            BRANCH_A,
            downsample=16,
            output_dir=run_dir,
            epochs=3,
            batch_size=32,
            lr=1e-3,
            val_ratio=0.1,
            seed=SEED,
            model_name=f"fusion_inception_patchx_{name}",
            patience=4,
            branch_b_channels=BRANCH_B,
            fusion_model="inception_patchx",
            ablation_tags=tags,
        )
        rows.append(
            {
                "variant": name,
                "ablation_tags": sorted(tags),
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "nar": result["nar"],
                "fnr": result["fnr"],
                "latency_ms_per_sample": result["latency_ms_per_sample"],
            }
        )

    rows.sort(key=lambda item: (-float(item["accuracy"]), str(item["variant"])))
    full_row = next(row for row in rows if row["variant"] == "full")

    csv_lines = ["variant,accuracy,f1_macro,nar,fnr,latency_ms_per_sample,delta_acc,delta_f1"]
    md_lines = [
        "# Fusion Inception PatchX Ablation",
        "",
        "| Variant | Accuracy | F1 | NAR | FNR | Latency/ms | Delta Acc vs Full | Delta F1 vs Full |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta_acc = float(row["accuracy"]) - float(full_row["accuracy"])
        delta_f1 = float(row["f1_macro"]) - float(full_row["f1_macro"])
        csv_lines.append(
            "{variant},{accuracy:.6f},{f1:.6f},{nar:.6f},{fnr:.6f},{lat:.6f},{dacc:.6f},{df1:.6f}".format(
                variant=row["variant"],
                accuracy=float(row["accuracy"]),
                f1=float(row["f1_macro"]),
                nar=float(row["nar"]),
                fnr=float(row["fnr"]),
                lat=float(row["latency_ms_per_sample"]),
                dacc=delta_acc,
                df1=delta_f1,
            )
        )
        md_lines.append(
            "| {variant} | {accuracy:.4f} | {f1:.4f} | {nar:.4f} | {fnr:.4f} | {lat:.4f} | {dacc:+.4f} | {df1:+.4f} |".format(
                variant=row["variant"],
                accuracy=float(row["accuracy"]),
                f1=float(row["f1_macro"]),
                nar=float(row["nar"]),
                fnr=float(row["fnr"]),
                lat=float(row["latency_ms_per_sample"]),
                dacc=delta_acc,
                df1=delta_f1,
            )
        )

    (OUTPUT_DIR / "ablation_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "ablation_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "ablation_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    labels = [str(row["variant"]) for row in rows]
    acc_values = [float(row["accuracy"]) for row in rows]
    f1_values = [float(row["f1_macro"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(labels, acc_values, color="#1f77b4")
    axes[0].set_title("Ablation Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, f1_values, color="#ff7f0e")
    axes[1].set_title("Ablation F1")
    axes[1].set_ylabel("F1 Macro")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ablation_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved ablation study to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
