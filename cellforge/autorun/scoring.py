"""Score predictions on held-out perturbations with a fixed, harness-side metric.

The generated script never sees test expression. It receives the names of the
held-out perturbations and writes one predicted mean profile per perturbation
(log1p of counts per 10k). This module compares those profiles with the truth,
perturbation by perturbation, and scores two reference predictors on the same
perturbations: no change from control, and the mean shift seen in training.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..dataset import is_control


def _dense(matrix) -> np.ndarray:
    return np.asarray(matrix.todense() if hasattr(matrix, "todense") else matrix, dtype=np.float64)


def looks_like_counts(matrix) -> bool:
    values = matrix.data if hasattr(matrix, "nnz") else np.asarray(matrix).ravel()  # sparse keeps nonzeros in .data
    values = values[values != 0][:10000]
    return bool(values.size) and bool(np.allclose(values, np.round(values)))


def group_means(adata, key: str, genes: List[str]) -> Dict[str, np.ndarray]:
    """Mean log1p(CP10k) profile per label; library size uses all genes, not the subset."""

    counts = looks_like_counts(adata.X)
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    sub = adata.X[:, adata.var_names.get_indexer(genes)]
    labels = adata.obs[key].astype(str).to_numpy()
    means = {}
    for label in np.unique(labels):
        rows = labels == label
        block = _dense(sub[rows])
        if counts:
            block = np.log1p(block / np.maximum(totals[rows, None], 1) * 1e4)
        means[label] = block.mean(axis=0)
    return means


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _r2(pred: np.ndarray, truth: np.ndarray) -> float:
    total = ((truth - truth.mean()) ** 2).sum()
    return float(1 - ((truth - pred) ** 2).sum() / total) if total > 0 else 0.0


def profile_metrics(pred: np.ndarray, truth: np.ndarray, control: np.ndarray, n_de: int = 20) -> Dict[str, float]:
    de = np.argsort(-np.abs(truth - control))[:n_de]
    return {
        "MSE": float(((pred - truth) ** 2).mean()),
        "PCC": _pearson(pred, truth),
        "R2": _r2(pred, truth),
        "PCC_delta": _pearson(pred - control, truth - control),
        "MSE_DE": float(((pred[de] - truth[de]) ** 2).mean()),
        "PCC_DE": _pearson(pred[de], truth[de]),
        "R2_DE": _r2(pred[de], truth[de]),
    }


def score_predictions(
    predictions_path: str,
    test_path: str,
    train_path: str,
    key: str,
    n_de: int = 20,
) -> Dict[str, object]:
    import anndata as ad

    pred, test, train = ad.read_h5ad(predictions_path), ad.read_h5ad(test_path), ad.read_h5ad(train_path)
    genes = [g for g in pred.var_names if g in set(test.var_names) & set(train.var_names)]
    if not genes:
        raise ValueError("Predictions share no genes with the test set")
    truth = group_means(test, key, genes)
    train_means = group_means(train, key, genes)
    controls = [label for label in train_means if is_control(label)]
    if not controls:
        raise ValueError("Training split has no control cells to define the baseline profile")
    control = np.mean([train_means[c] for c in controls], axis=0)
    shifts = [m - control for label, m in train_means.items() if label not in controls]
    mean_shift = control + (np.mean(shifts, axis=0) if shifts else 0)

    pred_labels = pred.obs[key].astype(str).to_numpy()
    pred_matrix = _dense(pred[:, genes].X)
    rows = {"model": [], "no_change": [], "mean_shift": []}
    missing = []
    for label, observed in truth.items():
        hits = np.flatnonzero(pred_labels == label)
        if not len(hits):
            missing.append(label)
            continue
        rows["model"].append(profile_metrics(pred_matrix[hits[0]], observed, control, n_de))
        rows["no_change"].append(profile_metrics(control, observed, control, n_de))
        rows["mean_shift"].append(profile_metrics(mean_shift, observed, control, n_de))

    def average(entries: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
        return {m: float(np.mean([e[m] for e in entries])) for m in entries[0]} if entries else {}

    summary = {name: average(entries) for name, entries in rows.items()}
    model, baseline = summary["model"], summary["mean_shift"]
    return {
        "n_test_perturbations": len(truth),
        "n_scored": len(rows["model"]),
        "missing_perturbations": missing,
        "n_genes": len(genes),
        "n_de_genes": n_de,
        "metrics": summary,
        # a tie within float32 rounding is not a win
        "beats_mean_shift_on_PCC_delta": bool(model and model["PCC_delta"] > baseline["PCC_delta"] + 1e-6),
    }


def score_run(run_root: str) -> List[Dict[str, object]]:
    """Score every task in an autorun directory that wrote predictions."""

    summary_path = Path(run_root) / "autorun_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split = summary["split"]
    results = []
    for job in summary["jobs"]:
        predictions = Path(job["logs_dir"]) / "predictions.h5ad"
        if not predictions.exists():
            job["score"] = {"error": "no predictions.h5ad written"}
            continue
        job["score"] = score_predictions(str(predictions), split["test_h5ad"], split["train_h5ad"], split["perturbation_key"])
        (Path(job["logs_dir"]) / "test_scores.json").write_text(json.dumps(job["score"], indent=2), encoding="utf-8")
        results.append(job["score"])
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Score an autorun directory on its held-out perturbations.")
    parser.add_argument("run_root")
    args = parser.parse_args()
    for result in score_run(args.run_root):
        print(json.dumps(result["metrics"]["model"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
