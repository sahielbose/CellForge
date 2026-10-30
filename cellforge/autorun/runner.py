"""Autorun: split by perturbation, generate code per task, run it, and score it.

Splits follow the paper: perturbation identities, never cells, are divided
into five folds, and each run tests on one fold, so the score measures
prediction for perturbations the model has not seen. Validation perturbations
come from the remaining folds. Control cells always stay in training because
every prediction is made relative to them. Test expression is written for the
scorer only. With refinement on, a further share of perturbations becomes the
selection set: the script predicts it alongside the test set without knowing
which is which, and only the selection scores reach the refinement loop.
"""

import json
import os
import random
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..dataset import find_h5ad, is_control, perturbation_key
from ..paths import data_path, resolve_workspace_path, workspace_root
from .scoring import score_predictions


@dataclass
class JobRecord:
    task_id: str
    code_file: str
    status: str
    command: str
    logs_dir: str
    job_id: str = ""
    error: str = ""
    score: Optional[Dict[str, Any]] = None
    refinement: Optional[Dict[str, Any]] = None


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip()).strip("_").lower() or "unknown_dataset"


def prepare_split(dataset_file: Path, split_root: Path, n_folds=5, fold=0, val_ratio=0.1, seed=42,
                  select_ratio=0.0) -> Dict[str, Any]:
    manifest_path = split_root / "split_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    import anndata as ad

    adata = ad.read_h5ad(dataset_file)
    key = perturbation_key(adata.obs.columns)
    labels = adata.obs[key].astype(str)
    controls = sorted(label for label in labels.unique() if is_control(label))
    perturbations = sorted(label for label in labels.unique() if label not in controls)
    if not controls:
        raise ValueError(f"No control cells found in obs['{key}']; predictions need a control profile")
    if len(perturbations) < n_folds + 2:
        raise ValueError(f"Need at least {n_folds + 2} perturbations for {n_folds} folds; found {len(perturbations)}")
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}); got {fold}")

    random.Random(seed).shuffle(perturbations)
    test = sorted(perturbations[fold::n_folds])
    rest = [p for p in perturbations if p not in set(test)]
    n_val = max(1, round(len(perturbations) * val_ratio))
    n_select = max(1, round(len(perturbations) * select_ratio)) if select_ratio else 0
    val, select = sorted(rest[:n_val]), sorted(rest[n_val:n_val + n_select])
    train = rest[n_val + n_select:] + controls
    if len(train) == len(controls):
        raise ValueError("No perturbations left for training; lower the validation or selection share")

    split_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    groups = (("train", train), ("val", val), ("test", test)) + ((("select", select),) if select else ())
    for name, members in groups:
        paths[name] = split_root / f"{name}.h5ad"
        adata[labels.isin(members).to_numpy()].copy().write_h5ad(paths[name])
    (split_root / "test_perturbations.json").write_text(json.dumps(test), encoding="utf-8")
    (split_root / "val_perturbations.json").write_text(json.dumps(val), encoding="utf-8")
    # the script predicts the selection and test sets together and cannot tell them apart
    (split_root / "predict_perturbations.json").write_text(json.dumps(sorted(test + select)), encoding="utf-8")

    manifest = {
        "perturbation_key": key,
        "control_labels": controls,
        "n_train_perturbations": len(train) - len(controls),
        "n_val_perturbations": len(val),
        "n_test_perturbations": len(test),
        "n_select_perturbations": len(select),
        "n_folds": n_folds,
        "fold": fold,
        "val_ratio": val_ratio,
        "seed": seed,
        "train_h5ad": str(paths["train"]),
        "val_h5ad": str(paths["val"]),
        "test_h5ad": str(paths["test"]),
        "test_perturbations": str(split_root / "test_perturbations.json"),
        "predict_perturbations": str(split_root / "predict_perturbations.json"),
    }
    if select:
        manifest.update(select_ratio=select_ratio, select_h5ad=str(paths["select"]))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_command(code_file: Path, split: Dict[str, Any], logs_dir: Path, python: str = "python") -> str:
    """The CLI every generated script implements (see the acceptance contract)."""

    args = {
        "--adata-path": split["train_h5ad"],
        "--val-adata-path": split["val_h5ad"],
        "--output-metrics": logs_dir / "val_metrics.json",
        "--predict-perturbations": split.get("predict_perturbations", split["test_perturbations"]),
        "--predictions-out": logs_dir / "predictions.h5ad",
    }
    return " ".join([shlex.quote(python), shlex.quote(str(code_file))] + [f"{k} {shlex.quote(str(v))}" for k, v in args.items()])


def render_sbatch(job_name: str, command: str, logs_dir: Path, slurm: Dict[str, Any], env: Dict[str, str]) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={slurm['partition']}",
        f"#SBATCH --time={slurm['time']}",
        f"#SBATCH --cpus-per-task={slurm['cpus']}",
        f"#SBATCH --mem={slurm['mem']}",
        f"#SBATCH --output={logs_dir}/slurm-%j.out",
        f"#SBATCH --error={logs_dir}/slurm-%j.err",
    ]
    if slurm.get("gres"):
        lines.append(f"#SBATCH --gres={slurm['gres']}")
    lines += ["", "set -eo pipefail", f"cd {shlex.quote(str(workspace_root()))}"]
    if slurm.get("conda_env"):
        lines += ["source ~/.bashrc || true", f"conda activate {shlex.quote(slurm['conda_env'])}"]
    lines += [f"export {key}={shlex.quote(value)}" for key, value in env.items()]
    return "\n".join(lines + [command, ""])


def run_autorun(
    *,
    dataset_path: str,
    plans_dir: Optional[str] = None,
    max_tasks: Optional[int] = None,
    executor: str = "local",
    slurm: Optional[Dict[str, Any]] = None,
    n_folds: int = 5,
    fold: int = 0,
    split_val_ratio: float = 0.1,
    split_seed: int = 42,
    codegen_backend: str = "codex",
    timeout_hours: float = 8.0,
    approve: Optional[Callable[[str, str], bool]] = None,
    refine_rounds: int = 0,
    refine_branches: int = 1,
    refine_metric: str = "PCC_delta",
    flush_at: Sequence[float] = (),
    select_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    select_ratio = (0.1 if refine_rounds else 0.0) if select_ratio is None else select_ratio
    if refine_rounds and not select_ratio:
        raise ValueError("Refinement scores candidates on selection perturbations; set a selection share above 0")
    dataset_file = find_h5ad(str(resolve_workspace_path(dataset_path)))
    dataset_name = _slug(dataset_file.stem)
    plan_root = resolve_workspace_path(plans_dir) if plans_dir else data_path("plans", dataset_name)
    plans = sorted(plan_root.glob("research_plan_*.json"), key=lambda p: p.stat().st_mtime)
    if not plans:
        raise FileNotFoundError(f"No research plan in {plan_root}; run method design first")
    plan = json.loads(plans[-1].read_text(encoding="utf-8"))
    task_ids = plan_task_ids(plan)
    task_ids = task_ids[:max_tasks] if max_tasks else task_ids

    tag = f"fold{fold}of{n_folds}_val{round(split_val_ratio * 100)}_seed{split_seed}"
    tag += f"_sel{round(select_ratio * 100)}" if select_ratio else ""
    split = prepare_split(dataset_file, dataset_file.parent / "splits" / dataset_name / tag,
                          n_folds, fold, split_val_ratio, split_seed, select_ratio)
    run_root = data_path("runs", dataset_name, f"{datetime.now():%Y%m%d_%H%M%S}_{tag}")
    smoke_dir = smoke_set(dataset_file, plans[-1])
    env = {"CELLFORGE_PERTURBATION_KEY": split["perturbation_key"]}

    jobs: List[JobRecord] = []
    for task_id in task_ids:
        logs_dir = run_root / task_id / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        code_file = code_for(plan, plans[-1], task_id, codegen_backend, smoke_dir)
        if not code_file:
            jobs.append(JobRecord(task_id, "", "codegen_failed", "", str(logs_dir), error="code generation failed"))
            continue
        command = build_command(Path(code_file), split, logs_dir, _python(executor))
        if approve is not None and not approve(str(code_file), command):
            jobs.append(JobRecord(task_id, code_file, "not_approved", command, str(logs_dir)))
            continue

        def execute(script: str, logs: Path, wait: bool = False, task_id=task_id) -> JobRecord:
            logs.mkdir(parents=True, exist_ok=True)
            return _execute(task_id, script, build_command(Path(script), split, logs, _python(executor)), logs,
                            executor, slurm or {}, env, timeout_hours, f"cf_{dataset_name}_{task_id}", wait)

        if refine_rounds:
            from .refine import refine

            def improve(script: str, report: str, tag: str, task_id=task_id) -> Tuple[Optional[str], str]:
                from .. import Code_Generation

                scoped = task_id if plan.get("task_wise_plan") else None
                return Code_Generation.improve_code(plan, script, report, str(run_root / task_id / tag / "code"),
                                                    task_id=scoped, backend=codegen_backend, smoke_dir=smoke_dir)

            best, job, record = refine(
                code_file,
                run=lambda script, tag, task_id=task_id: execute(script, run_root / task_id / tag / "logs", wait=True),
                score=lambda job: _score(job, split["select_h5ad"], split, "selection"),
                improve=improve, rounds=refine_rounds, branches=refine_branches, metric=refine_metric,
                flush_at=flush_at,
            )
            job.refinement = record
        else:
            job = execute(code_file, logs_dir)
        if job.status == "completed":
            job.score = _score(job, split["test_h5ad"], split, "test")
        jobs.append(job)

    summary = {
        "dataset_file": str(dataset_file),
        "plan_file": str(plans[-1]),
        "run_root": str(run_root),
        "executor": executor,
        "split": split,
        "jobs": [asdict(job) for job in jobs],
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "autorun_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _python(executor: str) -> str:
    # local runs use this interpreter; Slurm jobs use the activated environment's python
    return sys.executable if executor == "local" else "python"


def _execute(task_id, code_file, command, logs_dir: Path, executor, slurm, env, timeout_hours, job_name,
             wait=False) -> JobRecord:
    if executor == "local":
        return _run_local(task_id, code_file, command, logs_dir, env, timeout_hours)
    script = logs_dir / "job.sbatch"
    script.write_text(render_sbatch(job_name, command, logs_dir, slurm, env))
    # --wait blocks until the job ends, which the refinement loop needs before it can score
    proc = subprocess.run(["sbatch", *(["--wait"] if wait else []), str(script)], text=True, capture_output=True)
    match = re.search(r"Submitted batch job (\d+)", proc.stdout)
    if proc.returncode == 0:
        status = "completed" if wait else "submitted"
    else:
        status = "failed" if wait and match else "submit_failed"
    return JobRecord(task_id, code_file, status, command, str(logs_dir),
                     job_id=match.group(1) if match else "", error=proc.stderr.strip())


def _score(job: JobRecord, truth_h5ad: str, split: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """Score the job's predictions for the perturbations in truth_h5ad and save them next to its logs."""

    predictions = Path(job.logs_dir) / "predictions.h5ad"
    if job.status != "completed" or not predictions.exists():
        return None
    result = score_predictions(str(predictions), truth_h5ad, split["train_h5ad"], split["perturbation_key"])
    (Path(job.logs_dir) / f"{name}_scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def smoke_set(dataset_file: Path, plan_file: Path) -> Optional[Path]:
    """Stand-in data for the verifier's smoke run, built once per dataset when CODEGEN_SMOKE_TEST is on."""

    if os.getenv("CODEGEN_SMOKE_TEST", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    from .smoke import build_smoke_set

    return build_smoke_set(dataset_file, data_path("codes", plan_file.parent.name, "smoke"))


def plan_task_ids(plan: Dict[str, Any]) -> List[str]:
    """The plan's task blocks, or one task covering the whole plan when it has none."""

    return [str(block["task_id"]) for block in plan.get("task_wise_plan", []) if block.get("task_id")] or ["T01"]


def code_for(plan: Dict[str, Any], plan_file: Path, task_id: str, backend: str,
             smoke_dir: Optional[Path] = None) -> Optional[str]:
    """Generate a task's script once per plan, so every fold trains the same design.

    The code generation phase and autorun share this cache under
    data/codes/<dataset>/<plan>/<task>.
    """

    from .. import Code_Generation

    code_dir = data_path("codes", plan_file.parent.name, plan_file.stem, task_id)
    record = code_dir / "script_path.txt"
    if record.exists() and Path(record.read_text(encoding="utf-8").strip()).exists():
        return record.read_text(encoding="utf-8").strip()
    scoped = task_id if plan.get("task_wise_plan") else None
    code_file = Code_Generation.generate_code_from_plan_task(plan, task_id=scoped, output_dir=str(code_dir),
                                                             backend=backend, smoke_dir=smoke_dir)
    if code_file:
        code_dir.mkdir(parents=True, exist_ok=True)
        record.write_text(str(code_file), encoding="utf-8")
    return code_file


def _run_local(task_id, code_file, command, logs_dir: Path, env, timeout_hours) -> JobRecord:
    clean_env = {k: v for k, v in os.environ.items() if not re.search(r"(API_KEY|TOKEN|SECRET)", k, re.I)}
    try:
        proc = subprocess.run(command, shell=True, cwd=workspace_root(), text=True, capture_output=True,
                              env={**clean_env, **env}, timeout=timeout_hours * 3600)
    except subprocess.TimeoutExpired:
        return JobRecord(task_id, code_file, "timeout", command, str(logs_dir), error=f"exceeded {timeout_hours} h")
    (logs_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (logs_dir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    status = "completed" if proc.returncode == 0 else "failed"
    return JobRecord(task_id, code_file, status, command, str(logs_dir),
                     error="" if proc.returncode == 0 else f"exit code {proc.returncode}")
