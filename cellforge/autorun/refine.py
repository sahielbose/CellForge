"""Closed-loop refinement on held-out selection perturbations (condition C1 in docs/EXPERIMENTS.md).

The split sets aside selection perturbations that the script is asked to
predict but never receives, mixed in with the test perturbations. Each round,
the coding agent gets the current best script, its selection scores against the
reference predictors, the tail of its training log and a journal of earlier
attempts, and makes one change. A candidate that fails in training gets one
debug turn with its error log. A candidate replaces the best script only if its
gain, paired by perturbation, exceeds both ``min_gain`` and ``z`` standard
errors, so a small selection set does not reward noise. With several branches,
each round tries that many changes, each shown the ones already tried, and keeps
the best (greedy best-of-k search). A flush point clears the journal, so later
rounds start from the best script without the history of earlier attempts, the
forgetting step of Fork-and-Flush. Test perturbations are scored once, for the
final script.
"""

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LOWER_IS_BETTER = {"MSE", "MSE_DE"}
REPORTED = ("PCC_delta", "PCC_delta_pert", "Discrimination", "MSE_DE", "PCC_DE", "Recall", "AUROC", "MSE")
REFERENCES = (("mean_shift", "mean shift"), ("linear", "linear"), ("no_change", "no change"))

Run = Callable[[str, str], Any]  # (script, tag) -> JobRecord
Score = Callable[[Any], Optional[Dict[str, Any]]]  # job -> selection scores, or None when it wrote no predictions
Improve = Callable[[str, str, str], Tuple[Optional[str], str]]  # (script, report, tag) -> (new script, summary)


def _log_tail(job, lines: int = 30) -> str:
    logs = sorted(Path(job.logs_dir).glob("*.out")) + [Path(job.logs_dir) / "stdout.log"]
    for path in logs:
        if path.exists() and path.stat().st_size:
            return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    return "(no training output)"


def feedback(score: Dict[str, Any], metric: str, journal: List[str], job) -> str:
    metrics = score["metrics"]
    columns = ["model"] + [key for key, _ in REFERENCES if metrics.get(key)]
    rows = "\n".join(
        f"{name:<15}" + "".join(f"{metrics[c][name]:>12.4f}" for c in columns)
        for name in REPORTED if name in metrics["model"]
    )
    header = f"{'metric':<15}" + "".join(f"{dict(REFERENCES, model='this script')[c]:>12}" for c in columns)
    direction = "lower" if metric in LOWER_IS_BETTER else "higher"
    return (
        f"SELECTION SCORES on {score['n_scored']} unseen perturbations (the target is {metric}, {direction} is better)\n"
        f"{header}\n{rows}\n\n"
        f"PREVIOUS ATTEMPTS\n{chr(10).join(journal) or '- none yet'}\n\n"
        f"END OF THE TRAINING LOG\n{_log_tail(job)}"
    )


def paired_gain(candidate: Dict[str, Any], best: Dict[str, Any], metric: str) -> Tuple[float, float]:
    """Mean improvement over the perturbations both scored, signed so that positive is better, and its standard error."""

    sign = -1 if metric in LOWER_IS_BETTER else 1
    a, b = candidate.get("per_perturbation", {}), best.get("per_perturbation", {})
    diffs = [sign * (a[p][metric] - b[p][metric]) for p in a.keys() & b.keys()]
    if not diffs:
        return sign * (candidate["metrics"]["model"][metric] - best["metrics"]["model"][metric]), math.inf
    mean = sum(diffs) / len(diffs)
    if len(diffs) < 2:
        return mean, 0.0
    variance = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    return mean, math.sqrt(variance / len(diffs))


def _debug_report(job) -> str:
    return ("The edited script did not finish training or wrote no predictions. Fix the failure while keeping the "
            f"change you made.\n\nERROR\n{job.error or 'none reported'}\n\nEND OF THE LOG\n{_log_tail(job, 60)}")


def refine(
    code_file: str,
    *,
    run: Run,
    score: Score,
    improve: Improve,
    rounds: int,
    branches: int = 1,
    metric: str = "PCC_delta",
    min_gain: float = 1e-3,
    z: float = 1.0,
    flush_at: Sequence[float] = (),
) -> Tuple[str, Any, Dict[str, Any]]:
    """Return the best script, its job (whose predictions include the test rows) and a record of the search."""

    best_file, best_job = code_file, run(code_file, "round00")
    best = score(best_job)
    record: Dict[str, Any] = {"metric": metric, "rounds": rounds, "branches": branches, "evaluations": 1, "attempts": []}
    if best is None:
        record["error"] = "the starting script wrote no predictions, so there is nothing to refine"
        return best_file, best_job, record
    if metric not in best["metrics"]["model"]:
        record["error"] = f"{metric} is not defined on this selection set (Discrimination needs two perturbations)"
        return best_file, best_job, record

    def evaluate(path: str, tag: str):
        job = run(path, tag)
        record["evaluations"] += 1
        return job, score(job)

    flush_rounds = {max(1, round(f * rounds)) for f in flush_at}
    journal: List[str] = []
    for r in range(1, rounds + 1):
        if r in flush_rounds:
            journal.clear()
        report = feedback(best, metric, journal, best_job)
        tried: List[Dict[str, Any]] = []
        for b in range(branches):
            tag = f"round{r:02d}" + (f"_branch{b}" if branches > 1 else "")
            siblings = "\n".join(f"- {t['summary'][:200]}" for t in tried)
            path, summary = improve(best_file, report + (
                f"\n\nCHANGES ALREADY TRIED THIS ROUND (make a different kind of change)\n{siblings}" if siblings else ""
            ), tag)
            attempt = {"tag": tag, "script": path, "summary": summary, "job": None, "score": None}
            if path:
                attempt["job"], attempt["score"] = evaluate(path, tag)
                if attempt["score"] is None:
                    fixed, _ = improve(path, _debug_report(attempt["job"]), tag + "_debug")
                    if fixed:
                        attempt["script"] = fixed
                        attempt["job"], attempt["score"] = evaluate(fixed, tag + "_debug")
            tried.append(attempt)

        current = best["metrics"]["model"][metric]
        for attempt in tried:
            if attempt["score"]:
                attempt["gain"], attempt["se"] = paired_gain(attempt["score"], best, metric)
        winner = max((t for t in tried if t["score"]), key=lambda t: t["gain"], default=None)
        keep = winner is not None and winner["gain"] > max(min_gain, z * winner["se"])
        for attempt in tried:
            value = attempt["score"]["metrics"]["model"][metric] if attempt["score"] else None
            if value is None:
                outcome = "failed verification" if not attempt["script"] else "wrote no predictions"
            else:
                outcome = "kept" if keep and attempt is winner else "discarded"
            shown = f"{metric} {value:.4f} against {current:.4f}" if value is not None else "no score"
            journal.append(f"- round {r}: {attempt['summary'][:300] or 'no description'} -> {shown}, {outcome}")
            record["attempts"].append({"tag": attempt["tag"], "script": attempt["script"], "summary": attempt["summary"],
                                       metric: value, "gain": attempt.get("gain"), "se": attempt.get("se"),
                                       "outcome": outcome})
        if keep:
            best_file, best_job, best = winner["script"], winner["job"], winner["score"]
    record["selection_score"] = best
    record["final_script"] = best_file
    return best_file, best_job, record
