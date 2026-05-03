#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import itertools
import pywt
import scipy.io as sio
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = {
    0: "background",
    1: "dig",
    2: "knock",
    3: "water",
    4: "shake",
    5: "walk",
}

SAMPLE_TENSOR_CACHE: dict[tuple[str, str, tuple[int, ...], int], torch.Tensor] = {}


@dataclass(frozen=True)
class SampleEntry:
    split: str
    relative_path: str
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fusion DAS benchmark pipeline")
    parser.add_argument(
        "--data-root", type=Path, default=Path("/Volumes/Data/das_data")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/das_project/outputs")
    )
    parser.add_argument("--branch-a", default="0,1,2,3,4,5")
    parser.add_argument("--branch-b", default="6,7,8,9,10,11")
    parser.add_argument("--feature-downsample", type=int, default=16)
    parser.add_argument("--cnn-downsample", type=int, default=16)
    parser.add_argument("--mpe-scales", default="1,2,3,4")
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--cnn-epochs", type=int, default=6)
    parser.add_argument("--cnn-batch-size", type=int, default=64)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimize-classical", action="store_true")
    parser.add_argument("--cnn-patience", type=int, default=4)
    parser.add_argument("--search-splits", action="store_true")
    parser.add_argument("--search-limit-per-class", type=int, default=40)
    parser.add_argument("--search-epochs", type=int, default=4)
    parser.add_argument("--max-search-candidates", type=int, default=12)
    parser.add_argument("--fusion-model", default="cross_attention")
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--cnn-only", action="store_true")
    parser.add_argument("--skip-innovative-models", action="store_true")
    parser.add_argument("--model-filter", default="")
    parser.add_argument("--ablation-tags", default="")
    return parser.parse_args()


def parse_channels(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def parse_model_filter(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def parse_ablation_tags(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split_entries(
    data_root: Path, split: str, limit_per_class: int = 0
) -> list[SampleEntry]:
    label_path = data_root / split / "label.txt"
    entries: list[SampleEntry] = []
    missing_paths = 0
    with label_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            relative_path, label_text = raw.split()
            normalized_path = relative_path.lstrip("/")
            mat_path = data_root / split / normalized_path
            if not mat_path.exists():
                missing_paths += 1
                continue
            entries.append(
                SampleEntry(
                    split=split,
                    relative_path=normalized_path,
                    label=int(label_text),
                )
            )
    if missing_paths:
        print(f"Skipped {missing_paths} missing files from split '{split}'")
    if limit_per_class <= 0:
        return entries
    limited: list[SampleEntry] = []
    counts: dict[int, int] = {}
    for entry in entries:
        count = counts.get(entry.label, 0)
        if count < limit_per_class:
            limited.append(entry)
            counts[entry.label] = count + 1
    return limited


def load_matrix(data_root: Path, entry: SampleEntry) -> np.ndarray:
    mat_path = data_root / entry.split / entry.relative_path
    last_error: OSError | None = None
    for _ in range(3):
        try:
            return sio.loadmat(mat_path)["data"].astype(np.float32)
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    if last_error is not None:
        raise last_error
    return sio.loadmat(mat_path)["data"].astype(np.float32)


def load_branch_tensor(
    data_root: Path,
    entry: SampleEntry,
    channels: list[int],
    downsample: int,
) -> torch.Tensor:
    key = (entry.split, entry.relative_path, tuple(channels), downsample)
    cached = SAMPLE_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached
    matrix = load_matrix(data_root, entry)
    x = standardize_channels(matrix[:, channels])
    x = downsample_matrix(x, downsample).T.astype(np.float32)
    tensor = torch.from_numpy(x)
    SAMPLE_TENSOR_CACHE[key] = tensor
    return tensor


def warm_branch_cache(
    data_root: Path,
    entries: list[SampleEntry],
    branch_sets: list[list[int]],
    downsample: int,
) -> None:
    total = len(entries) * len(branch_sets)
    seen = 0
    for entry in entries:
        for channels in branch_sets:
            load_branch_tensor(data_root, entry, channels, downsample)
            seen += 1
            if seen % 4000 == 0 or seen == total:
                print(f"Cache warmup {seen}/{total}")


def standardize_channels(matrix: np.ndarray) -> np.ndarray:
    centered = signal.detrend(matrix, axis=0, type="linear")
    mean = centered.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, keepdims=True) + 1e-6
    return (centered - mean) / std


def downsample_matrix(matrix: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return matrix
    return matrix[::factor]


def wavelet_denoise(
    sig: np.ndarray, wavelet: str = "db4", level: int = 3
) -> np.ndarray:
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745 if detail.size else 0.0
    threshold = sigma * math.sqrt(2.0 * math.log(max(sig.size, 2)))
    denoised = [coeffs[0]]
    denoised.extend(pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:])
    recon = pywt.waverec(denoised, wavelet)
    return recon[: sig.size]


def branch_signals(
    matrix: np.ndarray, channels: list[int], downsample: int
) -> tuple[np.ndarray, np.ndarray]:
    chosen = matrix[:, channels]
    normalized = standardize_channels(chosen)
    normalized = downsample_matrix(normalized, downsample)
    collapsed = normalized.mean(axis=1)
    collapsed = wavelet_denoise(collapsed)
    collapsed = (collapsed - collapsed.mean()) / (collapsed.std() + 1e-6)
    return normalized, collapsed.astype(np.float32)


def zero_crossing_rate(sig: np.ndarray) -> float:
    signs = np.signbit(sig)
    return float(np.mean(signs[1:] != signs[:-1]))


def permutation_entropy(sig: np.ndarray, order: int = 3, delay: int = 1) -> float:
    if sig.size < order * delay + 1:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(sig, order * delay)
    embedded = windows[:, ::delay]
    if embedded.shape[1] != order:
        return 0.0
    patterns = np.argsort(embedded, axis=1)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    pe = -(probs * np.log(probs + 1e-12)).sum()
    return float(pe / np.log(math.factorial(order)))


def multiscale_permutation_entropy(
    sig: np.ndarray, scales: Iterable[int]
) -> list[float]:
    values = []
    for scale in scales:
        if scale <= 1:
            coarse = sig
        else:
            trimmed = sig[: (sig.size // scale) * scale]
            coarse = trimmed.reshape(-1, scale).mean(axis=1) if trimmed.size else sig
        values.append(permutation_entropy(coarse))
    return values


def stft_band_energies(sig: np.ndarray) -> list[float]:
    nperseg = min(128, sig.size)
    noverlap = min(max(nperseg // 2, 1), max(nperseg - 1, 0))
    _, _, zxx = signal.stft(sig, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(zxx) ** 2
    if power.size == 0:
        return [0.0] * 4
    spectrum = power.mean(axis=1)
    bins = np.array_split(spectrum, 4)
    total = spectrum.sum() + 1e-6
    return [float(item.sum() / total) for item in bins]


def time_features(sig: np.ndarray) -> list[float]:
    rms = np.sqrt(np.mean(sig**2))
    peak = float(np.max(np.abs(sig)))
    crest = peak / (rms + 1e-6)
    return [
        float(np.mean(sig)),
        float(np.std(sig)),
        float(np.mean(np.abs(sig))),
        float(rms),
        peak,
        crest,
        float(np.nan_to_num(stats.skew(sig))),
        float(np.nan_to_num(stats.kurtosis(sig))),
    ]


def build_feature_names(scales: list[int]) -> list[str]:
    names: list[str] = []
    for prefix in ["a", "b"]:
        names.extend(f"time_{prefix}_{idx}" for idx in range(8))
        names.extend(f"stft_{prefix}_{idx}" for idx in range(4))
        names.extend(f"mpe_{prefix}_s{scale}" for scale in scales)
        names.append(f"zcr_{prefix}")
    names.extend(["fusion_corr", "fusion_energy_ratio", "fusion_diff_std"])
    return names


def feature_vector(
    matrix: np.ndarray,
    branch_a_channels: list[int],
    branch_b_channels: list[int],
    downsample: int,
    mpe_scales: list[int],
) -> np.ndarray:
    _, sig_a = branch_signals(matrix, branch_a_channels, downsample)
    _, sig_b = branch_signals(matrix, branch_b_channels, downsample)
    feat: list[float] = []
    for sig in [sig_a, sig_b]:
        feat.extend(time_features(sig))
        feat.extend(stft_band_energies(sig))
        feat.extend(multiscale_permutation_entropy(sig, mpe_scales))
        feat.append(zero_crossing_rate(sig))
    corr = (
        np.corrcoef(sig_a, sig_b)[0, 1] if sig_a.std() > 0 and sig_b.std() > 0 else 0.0
    )
    energy_ratio = float(np.sum(sig_a**2) / (np.sum(sig_b**2) + 1e-6))
    diff_std = float(np.std(sig_a - sig_b))
    feat.extend([float(np.nan_to_num(corr)), energy_ratio, diff_std])
    return np.asarray(feat, dtype=np.float32)


def select_columns(names: list[str], predicate: Callable[[str], bool]) -> list[int]:
    return [idx for idx, name in enumerate(names) if predicate(name)]


def prepare_feature_cache(
    data_root: Path,
    output_dir: Path,
    split: str,
    entries: list[SampleEntry],
    branch_a_channels: list[int],
    branch_b_channels: list[int],
    downsample: int,
    mpe_scales: list[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{split}_ds{downsample}_a{'-'.join(map(str, branch_a_channels))}_b{'-'.join(map(str, branch_b_channels))}_mpe{'-'.join(map(str, mpe_scales))}.npz"
    cache_path = output_dir / cache_name
    feature_names = build_feature_names(mpe_scales)
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["x"], cached["y"], feature_names
    x = np.zeros((len(entries), len(feature_names)), dtype=np.float32)
    y = np.zeros(len(entries), dtype=np.int64)
    for idx, entry in enumerate(entries):
        matrix = load_matrix(data_root, entry)
        x[idx] = feature_vector(
            matrix, branch_a_channels, branch_b_channels, downsample, mpe_scales
        )
        y[idx] = entry.label
        if (idx + 1) % 200 == 0 or idx + 1 == len(entries):
            print(f"[{split}] features {idx + 1}/{len(entries)}")
    np.savez_compressed(cache_path, x=x, y=y)
    return x, y, feature_names


def nar_fnr(
    y_true: np.ndarray, y_pred: np.ndarray, background_label: int = 0
) -> tuple[float, float]:
    bg_mask = y_true == background_label
    event_mask = ~bg_mask
    nar = (
        float(np.mean(y_pred[bg_mask] != background_label)) if np.any(bg_mask) else 0.0
    )
    fnr = (
        float(np.mean(y_pred[event_mask] == background_label))
        if np.any(event_mask)
        else 0.0
    )
    return nar, fnr


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latency_ms: float,
    class_names: list[str],
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "latency_ms_per_sample": float(latency_ms),
    }
    nar, fnr = nar_fnr(y_true, y_pred)
    result["nar"] = float(nar)
    result["fnr"] = float(fnr)
    if probabilities is not None:
        pmax = probabilities.max(axis=1)
        result["pmax_mean"] = float(np.mean(pmax))
        result["pmax_std"] = float(np.std(pmax))
        result["psigma_mean"] = float(np.mean(np.std(probabilities, axis=1)))
    result["report"] = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    return result


def save_confusion(
    cm: np.ndarray, class_names: list[str], path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_pca_projection(
    x: np.ndarray, y: np.ndarray, path: Path, title: str, seed: int
) -> None:
    pca = PCA(n_components=2, random_state=seed)
    proj = pca.fit_transform(StandardScaler().fit_transform(x))
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in sorted(CLASS_NAMES):
        mask = y == label
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=14,
            alpha=0.7,
            label=CLASS_NAMES[label],
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def format_params(best_params: dict[str, object] | None) -> str:
    if not best_params:
        return ""
    return "; ".join(f"{k}={v}" for k, v in sorted(best_params.items()))


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_classical_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    seed: int,
    optimize: bool,
) -> list[dict[str, object]]:
    class_names = [CLASS_NAMES[idx] for idx in sorted(CLASS_NAMES)]
    mz_cols = select_columns(
        feature_names, lambda name: name.startswith("mpe_") or name.startswith("zcr_")
    )
    stft_cols = select_columns(feature_names, lambda name: name.startswith("stft_"))
    all_cols = list(range(len(feature_names)))
    models: list[tuple[str, Pipeline, list[int], dict[str, list[object]] | None]] = [
        (
            "mpe_zcr_svm",
            Pipeline(
                [
                    ("scaler", MinMaxScaler()),
                    (
                        "svc",
                        SVC(kernel="rbf", class_weight="balanced", random_state=seed),
                    ),
                ]
            ),
            mz_cols,
            {"svc__C": [1, 5, 10], "svc__gamma": ["scale", 0.1, 0.01]},
        ),
        (
            "stft_svm",
            Pipeline(
                [
                    ("scaler", MinMaxScaler()),
                    (
                        "svc",
                        SVC(kernel="rbf", class_weight="balanced", random_state=seed),
                    ),
                ]
            ),
            stft_cols,
            {"svc__C": [1, 5, 10], "svc__gamma": ["scale", 0.1, 0.01]},
        ),
        (
            "mpe_zcr_rf",
            Pipeline(
                [
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=300,
                            n_jobs=-1,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            mz_cols,
            {"rf__n_estimators": [200, 300], "rf__max_depth": [None, 12, 20]},
        ),
        (
            "pca_knn",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "pca",
                        PCA(n_components=min(12, x_train.shape[1]), random_state=seed),
                    ),
                    ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance")),
                ]
            ),
            all_cols,
            {
                "pca__n_components": [8, 12, min(16, x_train.shape[1])],
                "knn__n_neighbors": [3, 5, 7],
            },
        ),
        (
            "mpe_zcr_psvm",
            Pipeline(
                [
                    ("scaler", MinMaxScaler()),
                    (
                        "svc",
                        SVC(
                            kernel="rbf",
                            probability=True,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            mz_cols,
            {"svc__C": [1, 5, 10], "svc__gamma": ["scale", 0.1, 0.01]},
        ),
        (
            "fusion_psvm",
            Pipeline(
                [
                    ("scaler", MinMaxScaler()),
                    (
                        "svc",
                        SVC(
                            kernel="rbf",
                            probability=True,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            all_cols,
            {"svc__C": [1, 5, 10], "svc__gamma": ["scale", 0.1, 0.01]},
        ),
    ]
    results: list[dict[str, object]] = []
    save_pca_projection(
        x_train[:, all_cols],
        y_train,
        output_dir / "fusion_feature_pca.png",
        "Fusion Feature PCA",
        seed,
    )
    save_pca_projection(
        x_train[:, mz_cols],
        y_train,
        output_dir / "mpe_zcr_pca.png",
        "MPE+ZCR PCA",
        seed,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for name, model, cols, param_grid in models:
        print(f"Training {name} ...")
        best_params = None
        if optimize and param_grid:
            search = GridSearchCV(
                model,
                param_grid=param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(x_train[:, cols], y_train)
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            model.fit(x_train[:, cols], y_train)
        started = time.perf_counter()
        y_pred = model.predict(x_test[:, cols])
        elapsed = time.perf_counter() - started
        probabilities = (
            model.predict_proba(x_test[:, cols])
            if hasattr(model, "predict_proba")
            else None
        )
        metrics = evaluate_predictions(
            y_test,
            y_pred,
            1000.0 * elapsed / max(len(y_test), 1),
            class_names,
            probabilities,
        )
        cm = confusion_matrix(y_test, y_pred, labels=list(sorted(CLASS_NAMES)))
        save_confusion(cm, class_names, output_dir / f"{name}_cm.png", name)
        with (output_dir / f"{name}_report.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        results.append(
            {
                "model": name,
                "best_params": best_params,
                "optimization": "grid_search" if best_params else "default",
                **{k: v for k, v in metrics.items() if k != "report"},
            }
        )
    return results


class DASDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        entries: list[SampleEntry],
        channels: list[int],
        downsample: int,
    ) -> None:
        self.data_root = data_root
        self.entries = entries
        self.channels = channels
        self.downsample = downsample
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        entry = self.entries[index]
        x = load_branch_tensor(self.data_root, entry, self.channels, self.downsample)
        sample = (x, torch.tensor(entry.label, dtype=torch.long))
        self._cache[index] = sample
        return sample


class FusionDASDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        entries: list[SampleEntry],
        branch_a_channels: list[int],
        branch_b_channels: list[int],
        downsample: int,
    ) -> None:
        self.data_root = data_root
        self.entries = entries
        self.branch_a_channels = branch_a_channels
        self.branch_b_channels = branch_b_channels
        self.downsample = downsample
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        entry = self.entries[index]
        xa = load_branch_tensor(
            self.data_root, entry, self.branch_a_channels, self.downsample
        )
        xb = load_branch_tensor(
            self.data_root, entry, self.branch_b_channels, self.downsample
        )
        y = torch.tensor(entry.label, dtype=torch.long)
        sample = (xa, xb, y)
        self._cache[index] = sample
        return sample


class SmallDASCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class InceptionBlock1D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, use_bottleneck: bool = True
    ) -> None:
        super().__init__()
        bottleneck_channels = max(8, out_channels // 2)
        if use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Sequential(
                nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(bottleneck_channels),
                nn.ReLU(inplace=True),
            )
            branch_channels = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            branch_channels = in_channels
        each_out = max(8, out_channels // 4)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    branch_channels, each_out, kernel_size=9, padding=4, bias=False
                ),
                nn.Conv1d(
                    branch_channels, each_out, kernel_size=19, padding=9, bias=False
                ),
                nn.Conv1d(
                    branch_channels, each_out, kernel_size=39, padding=19, bias=False
                ),
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, each_out, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(each_out * 4)
        self.relu = nn.ReLU(inplace=True)
        self.project = (
            nn.Identity()
            if each_out * 4 == out_channels
            else nn.Conv1d(each_out * 4, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        merged = torch.cat([branch(z) for branch in self.branches] + [self.pool_branch(x)], dim=1)
        return self.relu(self.project(self.bn(merged)))


class ResidualInceptionStage(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.block1 = InceptionBlock1D(in_channels, hidden_dim)
        self.block2 = InceptionBlock1D(hidden_dim, hidden_dim)
        self.block3 = InceptionBlock1D(hidden_dim, hidden_dim)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block3(self.block2(self.block1(x)))
        return self.relu(out + self.shortcut(x))


class InceptionTimeClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResidualInceptionStage(hidden_dim, hidden_dim)
        self.stage2 = ResidualInceptionStage(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stage2(self.stage1(self.stem(x)))
        return self.classifier(feat)


class InceptionPatchEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        patch_len: int = 16,
        stride: int = 8,
        heads: int = 4,
        ablation_tags: set[str] | None = None,
    ) -> None:
        super().__init__()
        tags = ablation_tags or set()
        self.use_inception = "no_inception" not in tags
        self.use_patch = "no_patch" not in tags
        self.use_mix_gate = "no_mix_gate" not in tags
        self.patch_len = patch_len
        self.patch_stride = stride
        self.local_stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResidualInceptionStage(hidden_dim, hidden_dim)
        self.stage2 = ResidualInceptionStage(hidden_dim, hidden_dim)
        self.patch_encoder = PatchBranchEncoder(
            in_channels=hidden_dim,
            embed_dim=hidden_dim,
            patch_len=patch_len,
            stride=stride,
            layers=2,
            heads=heads,
        )
        self.mix_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local_feat = self.local_stem(x)
        if self.use_inception:
            local_feat = self.stage2(self.stage1(local_feat))
        local_pool = local_feat.mean(dim=-1)
        if self.use_patch:
            patch_seq, patch_pool = self.patch_encoder(local_feat)
        else:
            pooled_feat = F.avg_pool1d(
                local_feat,
                kernel_size=self.patch_len,
                stride=self.patch_stride,
                ceil_mode=False,
            )
            if pooled_feat.size(-1) == 0:
                pooled_feat = local_feat.mean(dim=-1, keepdim=True)
            patch_seq = pooled_feat.transpose(1, 2)
            patch_pool = pooled_feat.mean(dim=-1)
        if self.use_mix_gate:
            gate = self.mix_gate(torch.cat([local_pool, patch_pool], dim=1))
            pooled = gate * patch_pool + (1.0 - gate) * local_pool
        else:
            pooled = 0.5 * (local_pool + patch_pool)
        return patch_seq, pooled, local_pool, patch_pool


class BranchEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        score = torch.softmax(self.temporal_attn(feat), dim=-1)
        pooled = torch.sum(feat * score, dim=-1)
        return feat, pooled


class DualBranchAttentionCNN(nn.Module):
    def __init__(
        self, in_channels_a: int, in_channels_b: int, num_classes: int
    ) -> None:
        super().__init__()
        self.encoder_a = BranchEncoder(in_channels_a)
        self.encoder_b = BranchEncoder(in_channels_b)
        fusion_dim = 128
        self.branch_gate = nn.Sequential(
            nn.Linear(fusion_dim * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        self.channel_attn = nn.Sequential(
            nn.Linear(fusion_dim * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, fusion_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.head_a = nn.Linear(fusion_dim, num_classes)
        self.head_b = nn.Linear(fusion_dim, num_classes)
        self.gate_head = nn.Sequential(
            nn.Linear(fusion_dim * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(
        self, xa: torch.Tensor, xb: torch.Tensor, return_parts: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, pa = self.encoder_a(xa)
        _, pb = self.encoder_b(xb)
        diff = torch.abs(pa - pb)
        prod = pa * pb
        context = torch.cat([pa, pb, diff, prod], dim=1)
        gate_logits = self.branch_gate(context)
        gate = torch.softmax(gate_logits, dim=1)
        fused = gate[:, :1] * pa + gate[:, 1:] * pb
        channel = self.channel_attn(context)
        fused = fused * channel + fused
        final = torch.cat([fused, pa, pb, diff], dim=1)
        fusion_logits = self.classifier(final)
        logits_a = self.head_a(pa)
        logits_b = self.head_b(pb)
        head_weight = torch.softmax(self.gate_head(context), dim=1)
        combined = (
            head_weight[:, 0:1] * fusion_logits
            + head_weight[:, 1:2] * logits_a
            + head_weight[:, 2:3] * logits_b
        )
        if return_parts:
            return fusion_logits, logits_a, logits_b
        return combined


class CrossAttentionFusionCNN(nn.Module):
    def __init__(
        self, in_channels_a: int, in_channels_b: int, num_classes: int
    ) -> None:
        super().__init__()
        self.encoder_a = BranchEncoder(in_channels_a)
        self.encoder_b = BranchEncoder(in_channels_b)
        dim = 128
        self.cross_ab = nn.MultiheadAttention(
            dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.cross_ba = nn.MultiheadAttention(
            dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.self_fusion = nn.MultiheadAttention(
            dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.norm_f = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )
        self.head_a = nn.Linear(dim, num_classes)
        self.head_b = nn.Linear(dim, num_classes)
        self.head_f = nn.Linear(dim * 3, num_classes)

    def _pool(self, seq: torch.Tensor) -> torch.Tensor:
        score = torch.softmax(seq.mean(dim=-1, keepdim=True).transpose(1, 2), dim=-1)
        return torch.sum(seq * score.transpose(1, 2), dim=1)

    def forward(
        self, xa: torch.Tensor, xb: torch.Tensor, return_parts: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fa, pa = self.encoder_a(xa)
        fb, pb = self.encoder_b(xb)
        sa = fa.transpose(1, 2)
        sb = fb.transpose(1, 2)
        ab, _ = self.cross_ab(sa, sb, sb)
        ba, _ = self.cross_ba(sb, sa, sa)
        sa = self.norm_a(sa + ab)
        sb = self.norm_b(sb + ba)
        fusion_seq = torch.cat([sa, sb], dim=1)
        fusion_seq, _ = self.self_fusion(fusion_seq, fusion_seq, fusion_seq)
        fusion_seq = self.norm_f(fusion_seq + self.ffn(fusion_seq))
        pooled_a = self._pool(sa)
        pooled_b = self._pool(sb)
        pooled_f = self._pool(fusion_seq)
        logits_a = self.head_a(pooled_a)
        logits_b = self.head_b(pooled_b)
        fusion_logits = self.head_f(
            torch.cat(
                [pooled_f, torch.abs(pooled_a - pooled_b), pooled_a * pooled_b], dim=1
            )
        )
        combined = 0.6 * fusion_logits + 0.2 * logits_a + 0.2 * logits_b
        if return_parts:
            return fusion_logits, logits_a, logits_b
        return combined


class PatchBranchEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 128,
        patch_len: int = 16,
        stride: int = 8,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_embed = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_len, stride=stride, bias=False
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(-1)
        if seq_len < self.patch_len:
            x = F.pad(x, (0, self.patch_len - seq_len))
        tokens = self.patch_embed(x).transpose(1, 2)
        positions = torch.linspace(
            0.0, 1.0, tokens.size(1), device=tokens.device, dtype=tokens.dtype
        ).view(1, -1, 1)
        tokens = self.norm(tokens + positions)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return encoded, pooled


class PatchTSTFusionNet(nn.Module):
    def __init__(
        self, in_channels_a: int, in_channels_b: int, num_classes: int, dim: int = 128
    ) -> None:
        super().__init__()
        self.encoder_a = PatchBranchEncoder(in_channels_a, embed_dim=dim)
        self.encoder_b = PatchBranchEncoder(in_channels_b, embed_dim=dim)
        self.cross_ab = nn.MultiheadAttention(
            dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.cross_ba = nn.MultiheadAttention(
            dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            ),
            num_layers=2,
        )
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.head_a = nn.Linear(dim, num_classes)
        self.head_b = nn.Linear(dim, num_classes)
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )
        self.head_f = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, num_classes),
        )

    def forward(
        self, xa: torch.Tensor, xb: torch.Tensor, return_parts: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sa, pa = self.encoder_a(xa)
        sb, pb = self.encoder_b(xb)
        ab, _ = self.cross_ab(sa, sb, sb)
        ba, _ = self.cross_ba(sb, sa, sa)
        sa = self.norm_a(sa + ab)
        sb = self.norm_b(sb + ba)
        fusion_seq = self.fusion_encoder(torch.cat([sa, sb], dim=1))
        pooled_f = fusion_seq.mean(dim=1)
        logits_a = self.head_a(pa)
        logits_b = self.head_b(pb)
        fusion_context = torch.cat([pooled_f, torch.abs(pa - pb), pa * pb, 0.5 * (pa + pb)], dim=1)
        fusion_logits = self.head_f(fusion_context)
        weights = torch.softmax(self.gate(fusion_context), dim=1)
        combined = (
            weights[:, 0:1] * fusion_logits
            + weights[:, 1:2] * logits_a
            + weights[:, 2:3] * logits_b
        )
        if return_parts:
            return fusion_logits, logits_a, logits_b
        return combined


class InceptionPatchCrossAttentionNet(nn.Module):
    def __init__(
        self,
        in_channels_a: int,
        in_channels_b: int,
        num_classes: int,
        dim: int = 128,
        ablation_tags: set[str] | None = None,
    ) -> None:
        super().__init__()
        tags = ablation_tags or set()
        self.use_cross_attention = "no_cross_attention" not in tags
        self.use_branch_head = "no_aux_heads" not in tags
        self.use_branch_bias = "no_branch_bias" not in tags
        self.encoder_a = InceptionPatchEncoder(
            in_channels_a, hidden_dim=dim, ablation_tags=tags
        )
        self.encoder_b = InceptionPatchEncoder(
            in_channels_b, hidden_dim=dim, ablation_tags=tags
        )
        self.cross_ab = nn.MultiheadAttention(
            dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.cross_ba = nn.MultiheadAttention(
            dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            ),
            num_layers=2,
        )
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.head_a = nn.Linear(dim, num_classes)
        self.head_b = nn.Linear(dim, num_classes)
        self.branch_confidence = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )
        nn.init.zeros_(self.branch_confidence[-1].weight)
        branch_bias = [1.25, 0.0, 0.0] if self.use_branch_bias else [0.0, 0.0, 0.0]
        self.branch_confidence[-1].bias.data = torch.tensor(
            branch_bias, dtype=torch.float32
        )
        self.head_f = nn.Sequential(
            nn.Linear(dim * 8, dim * 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_classes),
        )

    def forward(
        self, xa: torch.Tensor, xb: torch.Tensor, return_parts: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sa, pa, local_a, patch_a = self.encoder_a(xa)
        sb, pb, local_b, patch_b = self.encoder_b(xb)
        if self.use_cross_attention:
            ab, _ = self.cross_ab(sa, sb, sb)
            ba, _ = self.cross_ba(sb, sa, sa)
            sa = self.norm_a(sa + ab)
            sb = self.norm_b(sb + ba)
        fusion_seq = self.fusion_encoder(torch.cat([sa, sb], dim=1))
        pooled_f = fusion_seq.mean(dim=1)
        diff = torch.abs(pa - pb)
        prod = pa * pb
        mean_pair = 0.5 * (pa + pb)
        local_diff = torch.abs(local_a - local_b)
        patch_diff = torch.abs(patch_a - patch_b)
        fusion_context = torch.cat(
            [
                pooled_f,
                diff,
                prod,
                mean_pair,
                torch.abs(pooled_f - mean_pair),
                0.5 * (local_a + local_b),
                0.5 * (patch_a + patch_b),
                local_diff + patch_diff,
            ],
            dim=1,
        )
        fusion_logits = self.head_f(fusion_context)
        logits_a = self.head_a(0.5 * (pa + local_a))
        logits_b = self.head_b(0.5 * (pb + local_b))
        if self.use_branch_head:
            weights = torch.softmax(
                self.branch_confidence(torch.cat([pooled_f, pa, pb, diff], dim=1)),
                dim=1,
            )
            combined = (
                weights[:, 0:1] * fusion_logits
                + weights[:, 1:2] * logits_a
                + weights[:, 2:3] * logits_b
            )
        else:
            combined = fusion_logits
        if return_parts:
            return fusion_logits, logits_a, logits_b
        return combined


class FocalCrossEntropy(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha = self.alpha[targets]
        loss = -alpha * ((1.0 - target_probs) ** self.gamma) * target_log_probs
        return loss.mean()


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        samples_per_class: np.ndarray,
        beta: float = 0.999,
        gamma: float = 2.0,
        hard_ratio: float = 0.7,
    ) -> None:
        super().__init__()
        effective_num = 1.0 - np.power(beta, np.maximum(samples_per_class, 1.0))
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.mean()
        self.register_buffer("alpha", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma
        self.hard_ratio = hard_ratio

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha = self.alpha[targets]
        losses = -alpha * ((1.0 - target_probs) ** self.gamma) * target_log_probs
        keep = max(1, int(math.ceil(losses.numel() * self.hard_ratio)))
        topk = torch.topk(losses, k=keep, largest=True).values
        return topk.mean()


def split_entries(
    entries: list[SampleEntry], val_ratio: float, seed: int
) -> tuple[list[SampleEntry], list[SampleEntry]]:
    indices = np.arange(len(entries))
    labels = np.array([entry.label for entry in entries])
    class_count = len(np.unique(labels))
    val_size = max(int(round(len(entries) * val_ratio)), class_count)
    if val_size >= len(entries):
        val_size = max(class_count, len(entries) // 5)
    if val_size >= len(entries):
        val_size = class_count
    bincount = np.bincount(labels)
    use_stratify = labels if bincount.min() >= 2 else None
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, stratify=use_stratify
    )
    return [entries[idx] for idx in train_idx], [entries[idx] for idx in val_idx]


def evaluate_cnn_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    logits_list = []
    labels_list = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                xb, yb = batch
                logits = model(xb.to(device))
            else:
                xa, xb, yb = batch
                logits = model(xa.to(device), xb.to(device))
            logits_list.append(logits.cpu())
            labels_list.append(yb)
    elapsed = time.perf_counter() - started
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0).numpy()
    probabilities = torch.softmax(logits, dim=1).numpy()
    preds = probabilities.argmax(axis=1)
    latency_ms = 1000.0 * elapsed / max(len(labels), 1)
    return labels, preds, probabilities, latency_ms


def build_cnn_loaders(
    data_root: Path,
    train_entries: list[SampleEntry],
    val_entries: list[SampleEntry],
    test_entries: list[SampleEntry],
    channels: list[int] | None,
    downsample: int,
    batch_size: int,
    branch_b_channels: list[int] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if branch_b_channels is None:
        train_dataset = DASDataset(data_root, train_entries, channels or [], downsample)
        val_dataset = DASDataset(data_root, val_entries, channels or [], downsample)
        test_dataset = DASDataset(data_root, test_entries, channels or [], downsample)
    else:
        train_dataset = FusionDASDataset(
            data_root, train_entries, channels or [], branch_b_channels, downsample
        )
        val_dataset = FusionDASDataset(
            data_root, val_entries, channels or [], branch_b_channels, downsample
        )
        test_dataset = FusionDASDataset(
            data_root, test_entries, channels or [], branch_b_channels, downsample
        )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader


def collect_dual_branch_parts(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    fusion_list = []
    a_list = []
    b_list = []
    labels_list = []
    with torch.no_grad():
        for xa, xb, yb in loader:
            fusion_logits, logits_a, logits_b = model(
                xa.to(device), xb.to(device), return_parts=True
            )
            fusion_list.append(fusion_logits.cpu().numpy())
            a_list.append(logits_a.cpu().numpy())
            b_list.append(logits_b.cpu().numpy())
            labels_list.append(yb.numpy())
    return (
        np.concatenate(fusion_list, axis=0),
        np.concatenate(a_list, axis=0),
        np.concatenate(b_list, axis=0),
        np.concatenate(labels_list, axis=0),
    )


def tune_dual_branch_ensemble(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[tuple[float, float, float], float]:
    fusion_logits, logits_a, logits_b, labels = collect_dual_branch_parts(
        model, loader, device
    )
    combined_labels, combined_pred, _, _ = evaluate_cnn_model(model, loader, device)
    best_weights = (-1.0, -1.0, -1.0)
    best_acc = accuracy_score(combined_labels, combined_pred)
    for wf in np.linspace(0.2, 0.8, 7):
        for wa in np.linspace(0.1, 0.7, 7):
            wb = 1.0 - wf - wa
            if wb < 0.0 or wb > 0.7:
                continue
            logits = wf * fusion_logits + wa * logits_a + wb * logits_b
            pred = logits.argmax(axis=1)
            acc = accuracy_score(labels, pred)
            if acc > best_acc:
                best_acc = acc
                best_weights = (float(wf), float(wa), float(wb))
    return best_weights, float(best_acc)


def candidate_channel_splits(
    total_channels: int, max_candidates: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    rng = np.random.default_rng(seed)
    channels = list(range(total_channels))
    half = total_channels // 2
    candidates: list[tuple[list[int], list[int]]] = []

    def add_candidate(a: list[int], b: list[int]) -> None:
        key = (tuple(sorted(a)), tuple(sorted(b)))
        rev = (key[1], key[0])
        existing = {
            (tuple(x), tuple(y))
            for x, y in [(tuple(c[0]), tuple(c[1])) for c in candidates]
        }
        if key in existing or rev in existing:
            return
        candidates.append((sorted(a), sorted(b)))

    add_candidate(channels[:half], channels[half:])
    add_candidate(channels[::2], channels[1::2])
    add_candidate([0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11])
    add_candidate([0, 1, 3, 4, 6, 9], [2, 5, 7, 8, 10, 11])
    all_combos = list(itertools.combinations(channels, half))
    rng.shuffle(all_combos)
    for combo in all_combos:
        if len(candidates) >= max_candidates:
            break
        add_candidate(list(combo), [c for c in channels if c not in combo])
    while len(candidates) < max_candidates:
        perm = rng.permutation(channels).tolist()
        add_candidate(sorted(perm[:half]), sorted(perm[half:]))
    return candidates[:max_candidates]


def build_deep_model(
    model_name: str,
    in_channels_a: int,
    num_classes: int,
    in_channels_b: int | None = None,
    fusion_model: str = "cross_attention",
    ablation_tags: set[str] | None = None,
) -> tuple[nn.Module, str, str]:
    if in_channels_b is None:
        if "inception" in model_name:
            return (
                InceptionTimeClassifier(
                    in_channels=in_channels_a, num_classes=num_classes
                ),
                "inceptiontime_single",
                "inceptiontime",
            )
        return (
            SmallDASCNN(in_channels=in_channels_a, num_classes=num_classes),
            "single_branch_cnn",
            "adaptive_training",
        )
    if fusion_model == "dual_branch_attention":
        return (
            DualBranchAttentionCNN(
                in_channels_a=in_channels_a,
                in_channels_b=in_channels_b,
                num_classes=num_classes,
            ),
            "dual_branch_attention",
            "dual_branch_attention",
        )
    if fusion_model == "patchtst":
        return (
            PatchTSTFusionNet(
                in_channels_a=in_channels_a,
                in_channels_b=in_channels_b,
                num_classes=num_classes,
            ),
            "patchtst_cross_attention",
            "patchtst_cross_attention",
        )
    if fusion_model == "inception_patchx":
        return (
            InceptionPatchCrossAttentionNet(
                in_channels_a=in_channels_a,
                in_channels_b=in_channels_b,
                num_classes=num_classes,
                ablation_tags=ablation_tags,
            ),
            "inception_patch_cross_attention",
            "inception_patch_cross_attention",
        )
    return (
        CrossAttentionFusionCNN(
            in_channels_a=in_channels_a,
            in_channels_b=in_channels_b,
            num_classes=num_classes,
        ),
        "cross_attention_fusion",
        "cross_attention_fusion",
    )


def build_optimizer_for_model(
    model: nn.Module, architecture_name: str, lr: float
) -> tuple[torch.optim.Optimizer, float, str]:
    if architecture_name == "inception_patch_cross_attention":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr * 0.5, weight_decay=5e-5
        )
        grad_clip = 0.75
        optimizer_name = "AdamW"
    elif "patch" in architecture_name:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        grad_clip = 1.0
        optimizer_name = "AdamW"
    elif "inception" in architecture_name:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr * 0.8, weight_decay=5e-5
        )
        grad_clip = 2.0
        optimizer_name = "AdamW"
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        grad_clip = 0.0
        optimizer_name = "Adam"
    return optimizer, grad_clip, optimizer_name


def build_scheduler_for_model(
    optimizer: torch.optim.Optimizer, architecture_name: str, epochs: int
) -> tuple[torch.optim.lr_scheduler.LRScheduler, str]:
    if architecture_name == "inception_patch_cross_attention":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs, 4), eta_min=1e-5
            ),
            "cosine",
        )
    return (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        ),
        "plateau",
    )


def filter_deep_runs(
    deep_runs: list[tuple[str, list[int], list[int] | None, str]],
    selected_models: set[str],
) -> list[tuple[str, list[int], list[int] | None, str]]:
    if not selected_models:
        return deep_runs
    return [row for row in deep_runs if row[0] in selected_models]


def train_cnn(
    data_root: Path,
    train_entries: list[SampleEntry],
    test_entries: list[SampleEntry],
    channels: list[int],
    downsample: int,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    seed: int,
    model_name: str,
    patience: int,
    branch_b_channels: list[int] | None = None,
    fusion_model: str = "cross_attention",
    ablation_tags: set[str] | None = None,
) -> dict[str, object]:
    class_names = [CLASS_NAMES[idx] for idx in sorted(CLASS_NAMES)]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_subset, val_subset = split_entries(train_entries, val_ratio, seed)
    train_loader, val_loader, test_loader = build_cnn_loaders(
        data_root,
        train_subset,
        val_subset,
        test_entries,
        channels,
        downsample,
        batch_size,
        branch_b_channels,
    )
    device = get_torch_device()
    model, architecture_name, optimization_name = build_deep_model(
        model_name=model_name,
        in_channels_a=len(channels),
        in_channels_b=len(branch_b_channels) if branch_b_channels is not None else None,
        num_classes=len(CLASS_NAMES),
        fusion_model=fusion_model,
        ablation_tags=ablation_tags,
    )
    model = model.to(device)
    print(f"Training {model_name} on {device.type}")
    train_labels = np.array([entry.label for entry in train_subset])
    counts = np.bincount(train_labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    if branch_b_channels is not None:
        class_weights[0] *= 1.8
        class_weights = class_weights / class_weights.mean()
    optimizer, grad_clip, optimizer_name = build_optimizer_for_model(
        model, architecture_name, lr
    )
    scheduler, scheduler_name = build_scheduler_for_model(
        optimizer, architecture_name, epochs
    )
    class_weight_tensor = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )
    label_smoothing = 0.02 if architecture_name == "inception_patch_cross_attention" else 0.0
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor, label_smoothing=label_smoothing
    )
    focal_criterion = FocalCrossEntropy(alpha=class_weight_tensor, gamma=2.0)
    cb_focal_criterion = ClassBalancedFocalLoss(
        samples_per_class=counts, gamma=2.0, hard_ratio=0.7
    ).to(device)
    fusion_arch = architecture_name == "inception_patch_cross_attention"
    aux_weight = 0.05 if fusion_arch else 0.15
    ce_weight = 0.5 if fusion_arch else 0.35
    focal_weight = 0.2 if fusion_arch else 0.25
    cb_weight = 0.3 if fusion_arch else 0.4
    background_penalty_weight = 0.1 if fusion_arch else 0.25
    best_state = None
    best_val = -1.0
    wait = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            if len(batch) == 2:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                aux_loss = torch.tensor(0.0, device=device)
            else:
                xa, xb, yb = batch
                xa = xa.to(device)
                xb = xb.to(device)
                yb = yb.to(device)
                fusion_logits, logits_a, logits_b = model(xa, xb, return_parts=True)
                logits = model(xa, xb)
                aux_loss = aux_weight * (criterion(logits_a, yb) + criterion(logits_b, yb))
            optimizer.zero_grad()
            ce_loss = criterion(logits, yb)
            focal_loss = focal_criterion(logits, yb)
            cb_focal_loss = cb_focal_criterion(logits, yb)
            penalty = torch.tensor(0.0, device=device)
            if branch_b_channels is not None:
                background_mask = yb == 0
                if torch.any(background_mask):
                    non_background_prob = torch.softmax(logits[background_mask], dim=1)[
                        :, 1:
                    ].sum(dim=1)
                    penalty = background_penalty_weight * non_background_prob.mean()
            loss = (
                ce_weight * ce_loss
                + focal_weight * focal_loss
                + cb_weight * cb_focal_loss
                + aux_loss
                + penalty
            )
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            batch_size_now = yb.size(0)
            running_loss += loss.item() * batch_size_now
            seen += batch_size_now
        val_true, val_pred, _, _ = evaluate_cnn_model(model, val_loader, device)
        val_acc = accuracy_score(val_true, val_pred)
        if scheduler_name == "plateau":
            scheduler.step(val_acc)
        else:
            scheduler.step()
        train_loss = running_loss / max(seen, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_acc": float(val_acc),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"{model_name} epoch {epoch}/{epochs} loss={train_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"{model_name} early stop at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    ensemble_weights = None
    val_ensemble_acc = None
    if branch_b_channels is None:
        y_true, y_pred, probabilities, latency_ms = evaluate_cnn_model(
            model, test_loader, device
        )
        metrics = evaluate_predictions(
            y_true, y_pred, latency_ms, class_names, probabilities
        )
    else:
        ensemble_weights, val_ensemble_acc = tune_dual_branch_ensemble(
            model, val_loader, device
        )
        started = time.perf_counter()
        if ensemble_weights[0] < 0.0:
            y_true, y_pred, probabilities, latency_ms = evaluate_cnn_model(
                model, test_loader, device
            )
        else:
            fusion_logits, logits_a, logits_b, y_true = collect_dual_branch_parts(
                model, test_loader, device
            )
            elapsed = time.perf_counter() - started
            wf, wa, wb = ensemble_weights
            combined_logits = wf * fusion_logits + wa * logits_a + wb * logits_b
            probabilities = torch.softmax(
                torch.from_numpy(combined_logits), dim=1
            ).numpy()
            y_pred = probabilities.argmax(axis=1)
            latency_ms = 1000.0 * elapsed / max(len(y_true), 1)
        metrics = evaluate_predictions(
            y_true, y_pred, latency_ms, class_names, probabilities
        )
    cm = confusion_matrix(y_true, y_pred, labels=list(sorted(CLASS_NAMES)))
    save_confusion(cm, class_names, output_dir / f"{model_name}_cm.png", model_name)
    with (output_dir / f"{model_name}_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    with (output_dir / f"{model_name}_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(
        output_dir / f"{model_name}_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
    )
    return {
        "model": model_name,
        "best_params": {
            "optimizer": optimizer_name,
            "gradient_clip": grad_clip,
            "scheduler": "CosineAnnealingLR" if scheduler_name == "cosine" else "ReduceLROnPlateau",
            "label_smoothing": label_smoothing,
            "class_weighting": True,
            "early_stopping_patience": patience,
            "epochs_run": len(history),
            "architecture": architecture_name,
            "ablation_tags": sorted(ablation_tags or []),
            "ensemble_weights": ensemble_weights,
            "val_ensemble_acc": val_ensemble_acc,
        },
        "optimization": optimization_name,
        **{k: v for k, v in metrics.items() if k != "report"},
    }


def write_summary(results: list[dict[str, object]], output_dir: Path) -> None:
    metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "nar",
        "fnr",
        "latency_ms_per_sample",
        "pmax_mean",
        "pmax_std",
        "psigma_mean",
    ]
    lines = [",".join(["model", *metrics])]
    for row in results:
        lines.append(
            ",".join(
                [str(row["model"]), *[str(row.get(metric, "")) for metric in metrics]]
            )
        )
    (output_dir / "benchmark_summary.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    sorted_results = sorted(
        results,
        key=lambda item: (-float(item.get("accuracy", 0.0)), str(item["model"])),
    )
    md_lines = [
        "# Benchmark Summary",
        "",
        "| Rank | Model | Group | Accuracy | F1 | NAR | FNR | Latency(ms) | Optimization | Best Params |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for idx, row in enumerate(sorted_results, start=1):
        model = str(row["model"])
        params_obj = row.get("best_params")
        params_text = format_params(params_obj) if isinstance(params_obj, dict) else ""
        if any(tag in model for tag in ["fusion_cnn", "fusion_patch", "patchtst", "fusion_inception"]):
            group = "Deep Fusion"
        elif any(tag in model for tag in ["cnn", "inception"]):
            group = "Deep Single"
        elif "psvm" in model:
            group = "Probabilistic"
        else:
            group = "Feature Engineered"
        md_lines.append(
            "| {rank} | {model} | {group} | {acc:.4f} | {f1:.4f} | {nar:.4f} | {fnr:.4f} | {lat:.4f} | {opt} | {params} |".format(
                rank=idx,
                model=model,
                group=group,
                acc=as_float(row.get("accuracy", 0.0)),
                f1=as_float(row.get("f1_macro", 0.0)),
                nar=as_float(row.get("nar", 0.0)),
                fnr=as_float(row.get("fnr", 0.0)),
                lat=as_float(row.get("latency_ms_per_sample", 0.0)),
                opt=row.get("optimization", "default"),
                params=params_text,
            )
        )
    (output_dir / "benchmark_summary.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )


def search_best_split(
    data_root: Path,
    output_dir: Path,
    seed: int,
    limit_per_class: int,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    patience: int,
    downsample: int,
    max_candidates: int,
    fusion_model: str,
    ablation_tags: set[str] | None = None,
) -> tuple[list[int], list[int], list[dict[str, object]]]:
    train_entries = load_split_entries(data_root, "train", limit_per_class)
    search_train_entries, search_eval_entries = split_entries(train_entries, val_ratio, seed)
    candidates = candidate_channel_splits(12, max_candidates, seed)
    search_dir = output_dir / "split_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    best_score = -1.0
    best_split = candidates[0]
    for idx, (branch_a, branch_b) in enumerate(candidates, start=1):
        candidate_dir = search_dir / f"candidate_{idx:02d}"
        result = train_cnn(
            data_root,
            search_train_entries,
            search_eval_entries,
            branch_a,
            downsample,
            candidate_dir,
            epochs,
            batch_size,
            lr,
            val_ratio,
            seed,
            f"fusion_search_{idx:02d}",
            patience,
            branch_b,
            fusion_model,
            ablation_tags,
        )
        score = as_float(result.get("accuracy", 0.0)) - 0.15 * as_float(
            result.get("nar", 0.0)
        )
        row = {
            "candidate": idx,
            "branch_a": branch_a,
            "branch_b": branch_b,
            "selection_split": "train_holdout_only",
            "search_train_size": len(search_train_entries),
            "search_eval_size": len(search_eval_entries),
            "score": score,
            **result,
        }
        results.append(row)
        if score > best_score:
            best_score = score
            best_split = (branch_a, branch_b)
    summary_path = search_dir / "split_search_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    lines = [
        "# Split Search Summary",
        "",
        "> Candidate ranking is computed on a training-internal holdout split only; the test set is not used during split selection.",
        "",
        "| Candidate | branch_a | branch_b | score | acc | f1 | nar | fnr |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    sorted_rows = sorted(results, key=lambda item: -as_float(item.get("score", 0.0)))
    for row in sorted_rows:
        lines.append(
            "| {candidate} | {a} | {b} | {score:.4f} | {acc:.4f} | {f1:.4f} | {nar:.4f} | {fnr:.4f} |".format(
                candidate=row["candidate"],
                a=row["branch_a"],
                b=row["branch_b"],
                score=as_float(row.get("score", 0.0)),
                acc=as_float(row.get("accuracy", 0.0)),
                f1=as_float(row.get("f1_macro", 0.0)),
                nar=as_float(row.get("nar", 0.0)),
                fnr=as_float(row.get("fnr", 0.0)),
            )
        )
    (search_dir / "split_search_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return best_split[0], best_split[1], results


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_models = parse_model_filter(args.model_filter)
    ablation_tags = parse_ablation_tags(args.ablation_tags)
    branch_a_channels = parse_channels(args.branch_a)
    branch_b_channels = parse_channels(args.branch_b)
    if args.search_splits:
        branch_a_channels, branch_b_channels, _ = search_best_split(
            args.data_root,
            args.output_dir,
            args.seed,
            args.search_limit_per_class,
            args.search_epochs,
            args.cnn_batch_size,
            args.cnn_lr,
            args.cnn_val_ratio,
            min(args.cnn_patience, args.search_epochs),
            args.cnn_downsample,
            args.max_search_candidates,
            args.fusion_model,
            ablation_tags,
        )
        print(
            f"Best split found: branch_a={branch_a_channels} branch_b={branch_b_channels}"
        )
    mpe_scales = [int(item) for item in args.mpe_scales.split(",") if item.strip()]
    train_entries = load_split_entries(args.data_root, "train", args.limit_per_class)
    test_entries = load_split_entries(args.data_root, "test", args.limit_per_class)
    results: list[dict[str, object]] = []
    if not args.cnn_only:
        x_train, y_train, feature_names = prepare_feature_cache(
            args.data_root,
            args.output_dir / "feature_cache",
            "train",
            train_entries,
            branch_a_channels,
            branch_b_channels,
            args.feature_downsample,
            mpe_scales,
        )
        x_test, y_test, _ = prepare_feature_cache(
            args.data_root,
            args.output_dir / "feature_cache",
            "test",
            test_entries,
            branch_a_channels,
            branch_b_channels,
            args.feature_downsample,
            mpe_scales,
        )
        results.extend(
            run_classical_models(
                x_train,
                y_train,
                x_test,
                y_test,
                feature_names,
                args.output_dir,
                args.seed,
                args.optimize_classical,
            )
        )
    if not args.classical_only:
        deep_runs: list[tuple[str, list[int], list[int] | None, str]] = [
            ("branch_a_cnn", branch_a_channels, None, "cross_attention"),
            ("branch_b_cnn", branch_b_channels, None, "cross_attention"),
            ("fusion_cnn", branch_a_channels, branch_b_channels, "cross_attention"),
        ]
        if not args.skip_innovative_models:
            deep_runs.extend(
                [
                    ("branch_a_inception", branch_a_channels, None, "cross_attention"),
                    ("branch_b_inception", branch_b_channels, None, "cross_attention"),
                    ("fusion_patchtst", branch_a_channels, branch_b_channels, "patchtst"),
                    (
                        "fusion_inception_patchx",
                        branch_a_channels,
                        branch_b_channels,
                        "inception_patchx",
                    ),
                ]
            )
        deep_runs = filter_deep_runs(deep_runs, selected_models)
        for model_name, model_channels, extra_channels, fusion_kind in deep_runs:
            set_seed(args.seed)
            results.append(
                train_cnn(
                    args.data_root,
                    train_entries,
                    test_entries,
                    model_channels,
                    args.cnn_downsample,
                    args.output_dir,
                    args.cnn_epochs,
                    args.cnn_batch_size,
                    args.cnn_lr,
                    args.cnn_val_ratio,
                    args.seed,
                    model_name,
                    args.cnn_patience,
                    extra_channels,
                    fusion_kind,
                    ablation_tags,
                )
            )
    write_summary(results, args.output_dir)
    print(f"Saved benchmark outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
