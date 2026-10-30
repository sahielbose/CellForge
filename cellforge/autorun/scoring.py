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

Two further metrics measure what is specific to each perturbation, because the
change from control is dominated by what all perturbations share (Viñas Torné et
al., Nature Biotechnology 2025). PCC_delta_pert correlates predicted and
observed change from the average training perturbation, on which the mean-shift
predictor scores 0 by construction. Discrimination asks whether a prediction is
closer, by L1 distance with the target genes left out, to its own perturbation's
observed effect than to the others in the set, as in the perturbation
discrimination score of the Arc Virtual Cell Challenge: 1 is perfect and a
constant prediction averages about 0.5.

Three reference predictors are scored on the same perturbations: no change from
control, the mean shift seen across training perturbations, and the linear model
of Ahlmann-Eltze, Huber and Anders (Nature Methods 2025). It embeds genes by PCA
of the training means and fits a ridge map from the embedding of a perturbed gene
to the change it causes, so it predicts any perturbation whose target genes are
measured; combinations add the changes of their parts, and perturbations with no
measured target (drugs, for example) fall back to the mean shift.

With de_test="wilcoxon" (or CELLFORGE_DE_TEST=wilcoxon) the DE set is ranked by
Scanpy's Wilcoxon rank-sum z score instead, as the paper does for its multimodal
readouts, against at most 2,000 control cells.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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


def _normalized(adata, key: str, genes: List[str], label: str, max_cells: Optional[int] = None) -> np.ndarray:
    """log1p(CP10k) cells of one label (CONTROL selects every control label), as group_stats computes them."""

    names = adata.obs[key].astype(str).to_numpy()
    rows = np.flatnonzero([is_control(n) for n in names] if label == CONTROL else names == label)
    if max_cells and len(rows) > max_cells:
        rows = np.sort(np.random.default_rng(0).choice(rows, max_cells, replace=False))
    block = _dense(adata.X[rows][:, adata.var_names.get_indexer(genes)])
    if looks_like_counts(adata.X):
        totals = np.asarray(adata.X[rows].sum(axis=1)).reshape(-1, 1)
        block = np.log1p(block / np.maximum(totals, 1) * 1e4)
    return block


def wilcoxon_z(perturbed: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Scanpy's rank_genes_groups(method="wilcoxon") score: rank-sum z per gene, no tie or continuity correction."""

    n, m = len(perturbed), len(control)
    ranks = rankdata(np.vstack([perturbed, control]), axis=0)[:n].sum(axis=0)
    z = (ranks - n * (n + m + 1) / 2) / np.sqrt(n * m * (n + m + 1) / 12)
    return np.nan_to_num(z)


def gene_targets(label: str, index: Dict[str, int]) -> List[int]:
    """Measured genes named in a perturbation label such as KLF1, KLF1+CEBPA or KLF1_CEBPA."""

    tokens = [t for t in re.split(r"[+_,;|/ ]+", label) if t]
    found = [index[t] for t in tokens if t in index]
    return found if len(found) == len(tokens) else []


def linear_baseline(train_stats: Dict[str, Stats], control: np.ndarray, genes: List[str], k: int = 10,
                    ridge: float = 0.1) -> Callable[[str], Optional[np.ndarray]]:
    """The linear model of Ahlmann-Eltze, Huber and Anders: Y = G W P^T + b, fit on single-gene perturbations.

    G holds the top k principal-component scores of the training means (genes by conditions, each condition
    centred over genes), P the rows of G for each perturbed gene, with a zero row for control, and
    W = (G^T G + ridge I)^-1 G^T (Y - b) P (P^T P + ridge I)^-1 on the change from control.
    """

    index = {gene: i for i, gene in enumerate(genes)}
    observed = {label: mean - control for label, (mean, _, _) in train_stats.items() if label != CONTROL}
    singles = {label: t[0] for label in observed if len(t := gene_targets(label, index)) == 1}
    if not singles:
        return lambda label: None
    conditions = np.column_stack([train_stats[label][0] for label in observed] + [control])
    centred = conditions - conditions.mean(axis=0)
    u, sv, _ = np.linalg.svd(centred, full_matrices=False)
    k = min(k, len(sv))
    G = u[:, :k] * sv[:k]
    Y = np.column_stack([observed[label] for label in singles] + [np.zeros_like(control)])
    P = np.vstack([G[singles[label]] for label in singles] + [np.zeros(k)])
    b = Y.mean(axis=1, keepdims=True)
    W = np.linalg.solve(G.T @ G + ridge * np.eye(k), G.T @ (Y - b) @ P) @ np.linalg.inv(P.T @ P + ridge * np.eye(k))
    by_gene = {genes[i]: label for label, i in singles.items()}

    def predict(label: str) -> Optional[np.ndarray]:
        targets = gene_targets(label, index)
        if not targets:
            return None
        parts = [observed[by_gene[genes[t]]] if genes[t] in by_gene else (b[:, 0] + G @ W @ G[t]) for t in targets]
        return control + np.sum(parts, axis=0)

    return predict


def discrimination(predicted: np.ndarray, observed: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    """Per perturbation, 1 - (expected zero-based rank of its own effect among all observed effects) / N."""

    n = len(observed)
    scores = np.empty(n)
    for i in range(n):
        distance = np.abs(observed[:, masks[i]] - predicted[i, masks[i]]).sum(axis=1)
        closer, tied = (distance < distance[i]).sum(), (distance == distance[i]).sum() - 1
        scores[i] = 1 - (closer + tied / 2) / n
    return scores


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


def _change(profile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """profile - reference, with differences within float32 rounding of the reference set to zero.

    Predictions are written as float32, so a script that predicts the reference exactly would otherwise
    show rounding noise as its change, and that noise can correlate with anything.
    """

    change = profile - reference
    change[np.abs(change) <= 4 * np.finfo(np.float32).eps * np.maximum(np.abs(reference), 1)] = 0
    return change


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


def profile_metrics(pred: np.ndarray, truth: np.ndarray, control: np.ndarray, de: np.ndarray,
                    pert_mean: Optional[np.ndarray] = None) -> Dict[str, float]:
    change = np.abs(pred - control)
    pert_mean = control if pert_mean is None else pert_mean
    positive = np.zeros(len(truth), dtype=bool)
    positive[de] = True
    return {
        "MSE": float(((pred - truth) ** 2).mean()),
        "PCC": _pearson(pred, truth),
        "R2": _r2(pred, truth),
        "PCC_delta": _pearson(pred - control, truth - control),
        "PCC_delta_pert": _pearson(_change(pred, pert_mean), truth - pert_mean),
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
    de_test: Optional[str] = None,
) -> Dict[str, object]:
    import anndata as ad

    de_test = de_test or os.getenv("CELLFORGE_DE_TEST", "t-test")
    if de_test not in {"t-test", "wilcoxon"}:
        raise ValueError(f"de_test must be t-test or wilcoxon, not {de_test}")
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
    mean_shift = control + (np.mean(shifts, axis=0) if shifts else 0)  # also the average perturbed profile
    linear = linear_baseline(train_stats, control, genes)
    control_cells = _normalized(train, key, genes, CONTROL, max_cells=2000) if de_test == "wilcoxon" else None

    pred_labels = pred.obs[key].astype(str).to_numpy() if pred is not None else None
    pred_matrix = _dense(pred[:, genes].X) if pred is not None else None
    labels, missing, profiles = [], [], {"model": [], "no_change": [], "mean_shift": [], "linear": []}
    rows = {name: [] for name in profiles}
    index = {gene: i for i, gene in enumerate(genes)}
    n_linear = 0
    for label, stats in truth.items():
        if label == CONTROL:
            continue
        hits = np.flatnonzero(pred_labels == label) if pred is not None else [None]
        if not len(hits):
            missing.append(label)
            continue
        if de_test == "wilcoxon":
            score = wilcoxon_z(_normalized(test, key, genes, label), control_cells)
            de = np.argsort(-np.abs(score), kind="stable")[:k]
        else:
            de = de_genes(stats, train_stats[CONTROL], k)
        linear_profile = linear(label)
        n_linear += linear_profile is not None
        candidates = {"no_change": control, "mean_shift": mean_shift,
                      "linear": mean_shift if linear_profile is None else linear_profile}
        if pred is not None:
            candidates["model"] = pred_matrix[hits[0]]
        labels.append(label)
        for name, profile in candidates.items():
            profiles[name].append(profile)
            rows[name].append(profile_metrics(profile, stats[0], control, de, mean_shift))

    if len(labels) > 1:  # discrimination needs at least two perturbations to tell apart
        observed = np.array([truth[label][0] - control for label in labels])
        masks = []
        for label in labels:
            mask = np.ones(len(genes), dtype=bool)
            mask[gene_targets(label, index)] = False
            masks.append(mask)
        for name, predicted in profiles.items():
            if predicted:
                for row, value in zip(rows[name], discrimination(np.array(predicted) - control, observed, masks),
                                      strict=True):
                    row["Discrimination"] = float(value)

    def average(entries: List[Dict[str, float]]) -> Dict[str, float]:
        return {m: float(np.mean([e[m] for e in entries])) for m in entries[0]} if entries else {}

    summary = {name: average(entries) for name, entries in rows.items()}
    model, baseline = summary["model"], summary["mean_shift"]
    return {
        "n_test_perturbations": len(labels) + len(missing),
        "n_scored": len(labels),
        "missing_perturbations": missing,
        "n_genes": len(genes),
        "n_de_genes": k,
        "de_test": de_test,
        "linear_coverage": n_linear / len(labels) if labels else 0.0,  # the rest fell back to the mean shift
        "metrics": summary,
        "per_perturbation": dict(zip(labels, rows["model"], strict=False)),  # the model's metrics per perturbation
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
    parser.add_argument("--de-test", choices=["t-test", "wilcoxon"], help="how to rank DE genes (default t-test)")
    args = parser.parse_args()
    if args.de_test:
        os.environ["CELLFORGE_DE_TEST"] = args.de_test
    if args.baselines:
        split = json.loads((Path(args.baselines) / "split_manifest.json").read_text(encoding="utf-8"))
        result = score_predictions(None, split["test_h5ad"], split["train_h5ad"], split["perturbation_key"])
        print(json.dumps({k: result["metrics"][k] for k in ("no_change", "mean_shift", "linear")}, indent=2))
        print(f"linear model covered {result['linear_coverage']:.0%} of perturbations; the rest used the mean shift")
        return 0
    if not args.run_root:
        parser.error("give an autorun folder or --baselines SPLIT_DIR")
    for result in score_run(args.run_root):
        print(json.dumps(result["metrics"]["model"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
