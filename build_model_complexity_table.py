#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from fusion_das_benchmark import CLASS_NAMES, build_deep_model


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "paper_b_tables"


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("branch_a_cnn", None, "cross_attention"),
        ("branch_b_cnn", None, "cross_attention"),
        ("branch_a_inception", None, "cross_attention"),
        ("branch_b_inception", None, "cross_attention"),
        ("fusion_patchtst", 6, "patchtst"),
        ("fusion_cnn", 6, "cross_attention"),
        ("fusion_inception_patchx", 6, "inception_patchx"),
    ]
    rows = []
    for model_name, in_channels_b, fusion_model in specs:
        model, architecture_name, _ = build_deep_model(
            model_name=model_name,
            in_channels_a=6,
            in_channels_b=in_channels_b,
            num_classes=len(CLASS_NAMES),
            fusion_model=fusion_model,
            ablation_tags=set(),
        )
        total, trainable = count_params(model)
        rows.append(
            {
                "model": model_name,
                "architecture": architecture_name,
                "params_total": total,
                "params_trainable": trainable,
                "params_million": total / 1_000_000.0,
            }
        )
    rows.sort(key=lambda item: item["params_total"])

    md_lines = [
        "# Model Complexity Table",
        "",
        "| Model | Architecture | Parameters | Trainable | Parameters (M) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {model} | {architecture} | {params_total} | {params_trainable} | {params_million:.3f} |".format(
                **row
            )
        )
    (OUT_DIR / "model_complexity_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (OUT_DIR / "model_complexity_table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved model complexity table.")


if __name__ == "__main__":
    main()
