"""Closed-loop refinement on held-out selection perturbations (condition C1 in docs/EXPERIMENTS.md).

The split sets aside selection perturbations that the script is asked to
predict but never receives, mixed in with the test perturbations. Each round,
the coding agent gets the current best script, its selection scores against the
two reference predictors, the tail of its training log and a journal of earlier
attempts, and makes one change. A candidate replaces the best script only if it
improves the selection metric by more than ``min_gain``. With several branches,
each round tries that many independent changes and keeps the best (greedy
best-of-k search). A flush point clears the journal, so later rounds start from
the best script without the history of earlier attempts, the forgetting step of
Fork-and-Flush. Test perturbations are scored once, for the final script.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LOWER_IS_BETTER = {"MSE", "MSE_DE"}
REPORTED = ("PCC_delta", "MSE_DE", "PCC_DE", "Recall", "AUROC", "MSE")

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
    rows = "\n".join(
        f"{name:<10} {metrics['model'][name]:>9.4f} {metrics['mean_shift'][name]:>11.4f} {metrics['no_change'][name]:>10.4f}"
        for name in REPORTED
    )
    direction = "lower" if metric in LOWER_IS_BETTER else "higher"
    return (
        f"SELECTION SCORES on {score['n_scored']} unseen perturbations (the target is {metric}, {direction} is better)\n"
        f"{'metric':<10} {'this script':>9} {'mean shift':>11} {'no change':>10}\n{rows}\n\n"
        f"PREVIOUS ATTEMPTS\n{chr(10).join(journal) or '- none yet'}\n\n"
        f"END OF THE TRAINING LOG\n{_log_tail(job)}"
    )


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
    flush_at: Sequence[float] = (),
) -> Tuple[str, Any, Dict[str, Any]]:
    """Return the best script, its job (whose predictions include the test rows) and a record of the search."""

    sign = -1 if metric in LOWER_IS_BETTER else 1
    best_file, best_job = code_file, run(code_file, "round00")
    best = score(best_job)
    record: Dict[str, Any] = {"metric": metric, "rounds": rounds, "branches": branches, "evaluations": 1, "attempts": []}
    if best is None:
        record["error"] = "the starting script wrote no predictions, so there is nothing to refine"
        return best_file, best_job, record

    flush_rounds = {max(1, round(f * rounds)) for f in flush_at}
    journal: List[str] = []
    for r in range(1, rounds + 1):
        if r in flush_rounds:
            journal.clear()
        current = best["metrics"]["model"][metric]
        report = feedback(best, metric, journal, best_job)
        tried = []
        for b in range(branches):
            tag = f"round{r:02d}" + (f"_branch{b}" if branches > 1 else "")
            path, summary = improve(best_file, report, tag)
            value, job, result = None, None, None
            if path:
                job = run(path, tag)
                record["evaluations"] += 1
                result = score(job)
                value = result["metrics"]["model"][metric] if result else None
            tried.append((value, path, job, result, summary, tag))

        scored = [t for t in tried if t[0] is not None]
        winner = max(scored, key=lambda t: sign * t[0], default=None)
        keep = winner is not None and sign * (winner[0] - current) > min_gain
        for value, path, _, _, summary, tag in tried:
            outcome = ("kept" if keep and value == winner[0] and path == winner[1] else "discarded") if value is not None \
                else ("failed verification" if not path else "wrote no predictions")
            shown = f"{metric} {value:.4f} against {current:.4f}" if value is not None else "no score"
            journal.append(f"- round {r}: {summary[:300] or 'no description'} -> {shown}, {outcome}")
            record["attempts"].append({"tag": tag, "script": path, "summary": summary, metric: value, "outcome": outcome})
        if keep:
            _, best_file, best_job, best, _, _ = winner
    record["selection_score"] = best
    record["final_script"] = best_file
    return best_file, best_job, record
