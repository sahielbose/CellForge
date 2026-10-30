"""Collect the test scores of autorun folds into one Markdown results page.

    python -m cellforge.autorun.report data/runs --out RESULTS-measured.md

Every autorun_summary.json under the given folders counts. Scores are grouped by
dataset and averaged over every trained model and fold, with the standard
deviation across them, next to the three reference predictors scored on the same
perturbations.
"""

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

METRICS = (("PCC_delta", "PCC on change from control"), ("PCC_delta_pert", "PCC on change from the mean perturbation"),
           ("Discrimination", "Discrimination"), ("MSE_DE", "MSE, DE genes"), ("PCC_DE", "PCC, DE genes"),
           ("Recall", "Recall@k"), ("AUROC", "AUROC"), ("PCC", "PCC"), ("MSE", "MSE"))
REFERENCES = (("linear", "Linear"), ("mean_shift", "Mean shift"), ("no_change", "No change"))


def collect(paths: Sequence[str]) -> Dict[str, List[Dict]]:
    """Scored jobs per dataset, each with its fold, plan, task and test score."""

    rows = defaultdict(list)
    for root in paths:
        for summary_file in sorted(Path(root).rglob("autorun_summary.json")):
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            for job in summary["jobs"]:
                score = job.get("score")
                if score and "metrics" in score:
                    rows[Path(summary["dataset_file"]).stem].append({
                        "fold": summary["split"]["fold"], "plan": summary["plan_file"], "task": job["task_id"],
                        "refined": bool(job.get("refinement")), "score": score,
                    })
    return rows


def _cell(values: List[float]) -> str:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return "n/a"
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}" if len(values) > 1 else f"{values[0]:.4f}"


def render(rows: Dict[str, List[Dict]]) -> str:
    lines = ["# Measured results", "",
             f"Collected by `python -m cellforge.autorun.report` on {date.today():%Y-%m-%d}. Every value is a mean over "
             "trained models and folds, with the standard deviation across them, scored on held-out test "
             "perturbations. MSE is lower is better and every other metric is higher is better.", ""]
    for dataset, jobs in sorted(rows.items()):
        folds = sorted({j["fold"] for j in jobs})
        wins = sum(j["score"]["beats_mean_shift_on_PCC_delta"] for j in jobs)
        coverage = np.mean([j["score"].get("linear_coverage", 0) for j in jobs])
        n_plans = len({j["plan"] for j in jobs})
        lines += [f"## {dataset}", "",
                  f"{len(jobs)} scored models from {n_plans} plan{'s' * (n_plans != 1)}, on folds "
                  f"{', '.join(map(str, folds))}. The designed model beat the mean shift on PCC_delta in {wins} of "
                  f"{len(jobs)}. The linear model covered {coverage:.0%} of test perturbations; the rest used the "
                  "mean shift.", "",
                  "| Metric | Designed model | " + " | ".join(name for _, name in REFERENCES) + " |",
                  "|---|" + "---:|" * (1 + len(REFERENCES))]
        for key, label in METRICS:
            cells = [_cell([j["score"]["metrics"][who].get(key) for j in jobs])
                     for who in ["model"] + [k for k, _ in REFERENCES]]
            if any(c != "n/a" for c in cells):
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += ["", "<details><summary>Per fold</summary>", "",
                  "| Fold | Task | Refined | PCC_delta | MSE_DE | AUROC | Mean shift PCC_delta |", "|---|---|---|---:|---:|---:|---:|"]
        for j in sorted(jobs, key=lambda j: (j["fold"], j["task"])):
            m, base = j["score"]["metrics"]["model"], j["score"]["metrics"]["mean_shift"]
            lines.append(f"| {j['fold']} | {j['task']} | {'yes' if j['refined'] else 'no'} | {m['PCC_delta']:.4f} | "
                         f"{m['MSE_DE']:.4f} | {m['AUROC']:.4f} | {base['PCC_delta']:.4f} |")
        lines += ["", "</details>", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="run folders to search for autorun_summary.json")
    parser.add_argument("--out", help="write the page here instead of printing it")
    args = parser.parse_args()
    rows = collect(args.paths)
    if not rows:
        print("No scored jobs found; score submitted Slurm jobs with python -m cellforge.autorun.scoring <run folder>")
        return 1
    page = render(rows)
    if args.out:
        Path(args.out).write_text(page, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
