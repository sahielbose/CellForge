"""Score predictions on held-out perturbations with the paper's metric definitions.

The generated script never sees test expression. It receives the names of the
held-out perturbations and writes one predicted mean profile per perturbation
(log1p of counts per 10k). Scores are perturbation-centric, as in the paper's
Methods: per held-out perturbation, compare mean profiles, then average.

The differentially expressed set for a perturbation is the top k genes by the
absolute t statistic against control cells, with Scanpy's
``t-test_overestim_var`` convention and k = min(50, p // 2, p - 1) for p genes.
Recall is Recovery@k (the overlap between the observed set and the k genes with
the largest predicted change from control, divided by k) and AUROC scores the
absolute predicted change against membership in the observed set.

Two reference predictors are scored on the same perturbations: no change from
control, and the mean shift seen across training perturbations.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata

from ..dataset import is_control

CONTROL = "__control__"
Stats = Tuple[np.ndarray, np.ndarray, int]  # mean, variance, number of cells


def _dense(matrix) -> np.ndarray:
    return np.asarray(matrix.todense() if hasattr(matrix, "todense") else matrix, dtype=np.float64)


def looks_like_counts(matrix) -> bool:
    values = matrix.data if hasattr(matrix, "nnz") else np.asarray(matrix).ravel()  # sparse keeps nonzeros in .data
    values = values[values != 0][:10000]
    return bool(values.size) and bool(np.allclose(values, np.round(values)))


def group_stats(adata, key: str, genes: List[str]) -> Dict[str, Stats]:
    """Mean and variance of log1p(CP10k) per label, with every control label pooled as one group.

    Library size uses all genes, not the scored subset.
    """

    counts = looks_like_counts(adata.X)
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    sub = adata.X[:, adata.var_names.get_indexer(genes)]
    names, index = np.unique(adata.obs[key].astype(str).to_numpy(), return_inverse=True)
    labels = np.array([CONTROL if is_control(name) else name for name in names], dtype=object)[index]
    stats = {}
    for label in np.unique(labels):
        rows = labels == label
        block = _dense(sub[rows])
        if counts:
            block = np.log1p(block / np.maximum(totals[rows, None], 1) * 1e4)
        n = int(rows.sum())
        stats[label] = (block.mean(axis=0), block.var(axis=0, ddof=1) if n > 1 else np.zeros(block.shape[1]), n)
    return stats


def group_means(adata, key: str, genes: List[str]) -> Dict[str, np.ndarray]:
    return {label: mean for label, (mean, _, _) in group_stats(adata, key, genes).items()}


def n_de_genes(n_genes: int) -> int:
    return max(1, min(50, n_genes // 2, n_genes - 1))


def de_genes(perturbed: Stats, control: Stats, k: int) -> np.ndarray:
    """Top k genes by |t|; t-test_overestim_var divides both variances by the perturbed group's size."""

    (mp, vp, n), (mc, vc, _) = perturbed, control
    t = (mp - mc) / np.sqrt((vp + vc) / max(n, 1) + 1e-12)
    return np.argsort(-np.abs(t), kind="stable")[:k]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _r2(pred: np.ndarray, truth: np.ndarray) -> float:
    total = ((truth - truth.mean()) ** 2).sum()
    return float(1 - ((truth - pred) ** 2).sum() / total) if total > 0 else 0.0


def _auroc(score: np.ndarray, positive: np.ndarray) -> float:
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return 0.5
    ranks = rankdata(score)  # ties get average ranks, so a constant score gives 0.5
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _recall(change: np.ndarray, positive: np.ndarray, k: int) -> float:
    """Recovery@k with ties at the cutoff counted by their expected share, so a flat prediction scores k/p."""

    cutoff = np.sort(change)[::-1][k - 1]
    above, tied = change > cutoff, change == cutoff
    hits = (above & positive).sum() + (k - above.sum()) * (tied & positive).sum() / tied.sum()
    return float(hits / k)


def profile_metrics(pred: np.ndarray, truth: np.ndarray, control: np.ndarray, de: np.ndarray) -> Dict[str, float]:
    change = np.abs(pred - control)
    positive = np.zeros(len(truth), dtype=bool)
    positive[de] = True
    return {
        "MSE": float(((pred - truth) ** 2).mean()),
        "PCC": _pearson(pred, truth),
        "R2": _r2(pred, truth),
        "PCC_delta": _pearson(pred - control, truth - control),
        "MSE_DE": float(((pred[de] - truth[de]) ** 2).mean()),
        "PCC_DE": _pearson(pred[de], truth[de]),
        "R2_DE": _r2(pred[de], truth[de]),
        "Recall": _recall(change, positive, len(de)),
        "AUROC": _auroc(change, positive),
    }


def score_predictions(
    predictions_path: Optional[str],
    test_path: str,
    train_path: str,
    key: str,
    n_de: Optional[int] = None,
) -> Dict[str, object]:
    import anndata as ad

    test, train = ad.read_h5ad(test_path), ad.read_h5ad(train_path)
    pred = ad.read_h5ad(predictions_path) if predictions_path else None  # None scores the baselines only
    genes = [g for g in (pred if pred is not None else test).var_names if g in set(test.var_names) & set(train.var_names)]
    if not genes:
        raise ValueError("Predictions share no genes with the test set")
    k = n_de or n_de_genes(len(genes))
    truth = group_stats(test, key, genes)
    train_stats = group_stats(train, key, genes)
    if CONTROL not in train_stats:
        raise ValueError("Training split has no control cells to define the baseline profile")
    control = train_stats[CONTROL][0]
    shifts = [mean - control for label, (mean, _, _) in train_stats.items() if label != CONTROL]
    mean_shift = control + (np.mean(shifts, axis=0) if shifts else 0)

    pred_labels = pred.obs[key].astype(str).to_numpy() if pred is not None else None
    pred_matrix = _dense(pred[:, genes].X) if pred is not None else None
    rows = {"model": [], "no_change": [], "mean_shift": []}
    missing = []
    for label, stats in truth.items():
        if label == CONTROL:
            continue
        hits = np.flatnonzero(pred_labels == label) if pred is not None else [None]
        if not len(hits):
            missing.append(label)
            continue
        de = de_genes(stats, train_stats[CONTROL], k)
        observed = stats[0]
        if pred is not None:
            rows["model"].append(profile_metrics(pred_matrix[hits[0]], observed, control, de))
        rows["no_change"].append(profile_metrics(control, observed, control, de))
        rows["mean_shift"].append(profile_metrics(mean_shift, observed, control, de))

    def average(entries: List[Dict[str, float]]) -> Dict[str, float]:
        return {m: float(np.mean([e[m] for e in entries])) for m in entries[0]} if entries else {}

    summary = {name: average(entries) for name, entries in rows.items()}
    model, baseline = summary["model"], summary["mean_shift"]
    return {
        "n_test_perturbations": len(rows["no_change"]) + len(missing),
        "n_scored": len(rows["no_change"]),
        "missing_perturbations": missing,
        "n_genes": len(genes),
        "n_de_genes": k,
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
    parser = argparse.ArgumentParser(description="Score held-out perturbations.")
    parser.add_argument("run_root", nargs="?", help="autorun folder whose jobs wrote predictions.h5ad")
    parser.add_argument("--baselines", metavar="SPLIT_DIR", help="score only the reference predictors on a split")
    args = parser.parse_args()
    if args.baselines:
        split = json.loads((Path(args.baselines) / "split_manifest.json").read_text(encoding="utf-8"))
        result = score_predictions(None, split["test_h5ad"], split["train_h5ad"], split["perturbation_key"])
        print(json.dumps({k: result["metrics"][k] for k in ("no_change", "mean_shift")}, indent=2))
        return 0
    if not args.run_root:
        parser.error("give an autorun folder or --baselines SPLIT_DIR")
    for result in score_run(args.run_root):
        print(json.dumps(result["metrics"]["model"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
