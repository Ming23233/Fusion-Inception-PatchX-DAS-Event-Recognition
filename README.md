# Fusion-Inception-PatchX DAS Event Recognition

This repository contains the experimental code used for DAS event recognition in the accompanying manuscript. The dataset used in the experiments is an external open-source/reference dataset and is not redistributed in this repository.

## Repository contents

- `fusion_das_benchmark.py`: main benchmark pipeline, including classical baselines, fusion CNN models, evaluation, and result export.
- `run_fusion_inception_ablation.py`: ablation experiments for the Fusion-Inception-PatchX model.
- `run_fair_epoch_curves.py`: fair epoch-budget comparison curves.
- `build_model_complexity_table.py`: model complexity table generation.
- `paperfig/`: scripts for generating architecture figures.
- `figures/`: selected manuscript figures.

Generated experiment folders, local data caches, model checkpoints, and LaTeX build files are excluded from version control.

## Data

The code expects the DAS dataset to be prepared locally with the following structure:

```text
<DATA_ROOT>/
  train/
    label.txt
    <class folders and .mat files>
  test/
    label.txt
    <class folders and .mat files>
```

Each line in `label.txt` should contain a relative `.mat` file path and its integer class label. Each `.mat` file should contain a `data` matrix.

For manuscript submission systems, select `Reference data` if the experimental data come from an existing open-source dataset. Use the original dataset title and citation in the submission form, and provide this GitHub repository as the code repository for reproducibility.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install the appropriate PyTorch build for your CUDA or CPU environment if needed:

```bash
pip install torch
```

## Reproducing Experiments

Run the main benchmark:

```bash
python fusion_das_benchmark.py \
  --data-root /path/to/das_data \
  --output-dir outputs_reproduce \
  --branch-a 0,2,4,6,8,10 \
  --branch-b 1,3,5,7,9,11 \
  --fusion-model inception_patchx \
  --cnn-epochs 6 \
  --cnn-batch-size 32 \
  --seed 42
```

For a quick smoke test, limit the number of samples per class:

```bash
python fusion_das_benchmark.py \
  --data-root /path/to/das_data \
  --output-dir outputs_smoke \
  --limit-per-class 10 \
  --cnn-epochs 1 \
  --seed 42
```

Run the ablation study:

```bash
python run_fusion_inception_ablation.py
```

Run fair epoch-budget curves:

```bash
python run_fair_epoch_curves.py
```

Note: `run_fusion_inception_ablation.py` and `run_fair_epoch_curves.py` currently use `DATA_ROOT = Path("/Volumes/Data/das_data")`. Edit that constant or run the main benchmark directly with `--data-root`.

## Outputs

The benchmark writes summary files and figures to the selected output directory, including:

- `benchmark_summary.csv`
- `benchmark_summary.json`
- `benchmark_summary.md`
- per-model reports, confusion matrices, and prediction files

These outputs are generated artifacts and are not required for the code repository upload.
