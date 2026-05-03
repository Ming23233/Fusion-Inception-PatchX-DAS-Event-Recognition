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
OUTPUT_DIR = ROOT / "outputs_fair_epoch_curves_v1"
DATA_ROOT = Path("/Volumes/Data/das_data")
BRANCH_A = [0, 2, 4, 6, 8, 10]
BRANCH_B = [1, 3, 5, 7, 9, 11]
SEED = 42
EPOCH_BUDGETS = [2, 4, 6, 8]


def write_outputs(rows: list[dict[str, object]], model_specs: list[tuple[str, str, set[str]]]) -> None:
    rows.sort(key=lambda item: (str(item["model"]), int(item["epochs"])))
    (OUTPUT_DIR / "fair_curve_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_lines = ["model,epochs,accuracy,f1_macro,nar,fnr,latency_ms_per_sample"]
    md_lines = [
        "# Fair Epoch Curves",
        "",
        "| Model | Epochs | Accuracy | F1 | NAR | FNR | Latency/ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        csv_lines.append(
            "{model},{epochs},{accuracy:.6f},{f1:.6f},{nar:.6f},{fnr:.6f},{lat:.6f}".format(
                model=row["model"],
                epochs=int(row["epochs"]),
                accuracy=float(row["accuracy"]),
                f1=float(row["f1_macro"]),
                nar=float(row["nar"]),
                fnr=float(row["fnr"]),
                lat=float(row["latency_ms_per_sample"]),
            )
        )
        md_lines.append(
            "| {model} | {epochs} | {accuracy:.4f} | {f1:.4f} | {nar:.4f} | {fnr:.4f} | {lat:.4f} |".format(
                model=row["model"],
                epochs=int(row["epochs"]),
                accuracy=float(row["accuracy"]),
                f1=float(row["f1_macro"]),
                nar=float(row["nar"]),
                fnr=float(row["fnr"]),
                lat=float(row["latency_ms_per_sample"]),
            )
        )
    (OUTPUT_DIR / "fair_curve_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "fair_curve_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    for metric, title, color_path in [
        ("accuracy", "Fair Accuracy vs Epoch Budget", OUTPUT_DIR / "fair_curve_accuracy.png"),
        ("f1_macro", "Fair F1 vs Epoch Budget", OUTPUT_DIR / "fair_curve_f1.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model_name, _, _ in model_specs:
            series = [row for row in rows if row["model"] == model_name]
            if not series:
                continue
            ax.plot(
                [int(row["epochs"]) for row in series],
                [float(row[metric]) for row in series],
                marker="o",
                linewidth=2,
                label=model_name,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch Budget")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(color_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


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

    model_specs = [
        ("fusion_cnn", "cross_attention", set()),
        ("fusion_inception_patchx", "inception_patchx", set()),
    ]

    rows: list[dict[str, object]] = []
    for epochs in EPOCH_BUDGETS:
        for model_name, fusion_model, ablation_tags in model_specs:
            run_dir = OUTPUT_DIR / f"{model_name}_e{epochs:02d}"
            report_path = run_dir / f"{model_name}_report.json"
            if report_path.exists():
                result = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                result = train_cnn(
                    DATA_ROOT,
                    train_entries,
                    test_entries,
                    BRANCH_A,
                    downsample=16,
                    output_dir=run_dir,
                    epochs=epochs,
                    batch_size=32,
                    lr=1e-3,
                    val_ratio=0.1,
                    seed=SEED,
                    model_name=model_name,
                    patience=epochs + 1,
                    branch_b_channels=BRANCH_B,
                    fusion_model=fusion_model,
                    ablation_tags=ablation_tags,
                )
            rows.append(
                {
                    "model": model_name,
                    "epochs": epochs,
                    "accuracy": result["accuracy"],
                    "f1_macro": result["f1_macro"],
                    "nar": result["nar"],
                    "fnr": result["fnr"],
                    "latency_ms_per_sample": result["latency_ms_per_sample"],
                }
            )
            write_outputs(rows, model_specs)

    print(f"Saved fair epoch curves to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
