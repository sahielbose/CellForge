"""Autorun: split by perturbation, generate code per task, run it, and score it.

Splits hold out whole perturbations, never cells, so test and validation
measure what the paper claims: prediction for perturbations the model has not
seen. Control cells always stay in training because every prediction is made
relative to them. Test expression is written for the scorer only.
"""

import json
import os
import random
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip()).strip("_").lower() or "unknown_dataset"


def prepare_split(dataset_file: Path, split_root: Path, test_ratio=0.2, val_ratio=0.1, seed=42) -> Dict[str, Any]:
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
    if len(perturbations) < 3:
        raise ValueError(f"Need at least 3 perturbations to split; found {len(perturbations)}")

    random.Random(seed).shuffle(perturbations)
    n_test = max(1, round(len(perturbations) * test_ratio))
    n_val = max(1, round(len(perturbations) * val_ratio))
    test, val = perturbations[:n_test], perturbations[n_test : n_test + n_val]
    train = perturbations[n_test + n_val :] + controls

    split_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, members in (("train", train), ("val", val), ("test", test)):
        paths[name] = split_root / f"{name}.h5ad"
        adata[labels.isin(members).to_numpy()].copy().write_h5ad(paths[name])
    (split_root / "test_perturbations.json").write_text(json.dumps(test), encoding="utf-8")
    (split_root / "val_perturbations.json").write_text(json.dumps(val), encoding="utf-8")

    manifest = {
        "perturbation_key": key,
        "control_labels": controls,
        "n_train_perturbations": len(train) - len(controls),
        "n_val_perturbations": len(val),
        "n_test_perturbations": len(test),
        "test_ratio": test_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "train_h5ad": str(paths["train"]),
        "val_h5ad": str(paths["val"]),
        "test_h5ad": str(paths["test"]),
        "test_perturbations": str(split_root / "test_perturbations.json"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_command(code_file: Path, split: Dict[str, Any], logs_dir: Path) -> str:
    """The CLI every generated script implements (see the acceptance contract)."""

    args = {
        "--adata-path": split["train_h5ad"],
        "--val-adata-path": split["val_h5ad"],
        "--output-metrics": logs_dir / "val_metrics.json",
        "--predict-perturbations": split["test_perturbations"],
        "--predictions-out": logs_dir / "predictions.h5ad",
    }
    return " ".join(["python", shlex.quote(str(code_file))] + [f"{k} {shlex.quote(str(v))}" for k, v in args.items()])


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
    split_test_ratio: float = 0.2,
    split_val_ratio: float = 0.1,
    split_seed: int = 42,
    codegen_backend: str = "codex",
    timeout_hours: float = 8.0,
) -> Dict[str, Any]:
    from ..Code_Generation import generate_code_from_plan_task

    dataset_file = find_h5ad(str(resolve_workspace_path(dataset_path)))
    dataset_name = _slug(dataset_file.stem)
    plan_root = resolve_workspace_path(plans_dir) if plans_dir else data_path("plans", dataset_name)
    plans = sorted(plan_root.glob("research_plan_*.json"), key=lambda p: p.stat().st_mtime)
    if not plans:
        raise FileNotFoundError(f"No research plan in {plan_root}; run method design first")
    plan = json.loads(plans[-1].read_text(encoding="utf-8"))
    task_ids = [block["task_id"] for block in plan.get("task_wise_plan", [])] or ["T01"]
    task_ids = task_ids[:max_tasks] if max_tasks else task_ids

    tag = f"test{round(split_test_ratio * 100)}_val{round(split_val_ratio * 100)}_seed{split_seed}"
    split = prepare_split(dataset_file, dataset_file.parent / "splits" / dataset_name / tag,
                          split_test_ratio, split_val_ratio, split_seed)
    run_root = data_path("runs", dataset_name, datetime.now().strftime("%Y%m%d_%H%M%S"))

    jobs: List[JobRecord] = []
    for task_id in task_ids:
        logs_dir = run_root / task_id / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        code_file = generate_code_from_plan_task(plan, task_id=task_id, output_dir=str(run_root / task_id / "code"),
                                                 backend=codegen_backend)
        if not code_file:
            jobs.append(JobRecord(task_id, "", "codegen_failed", "", str(logs_dir), error="code generation failed"))
            continue
        command = build_command(Path(code_file), split, logs_dir)
        env = {"CELLFORGE_PERTURBATION_KEY": split["perturbation_key"]}
        if executor == "local":
            job = _run_local(task_id, code_file, command, logs_dir, env, timeout_hours)
            if job.status == "completed" and (logs_dir / "predictions.h5ad").exists():
                job.score = score_predictions(str(logs_dir / "predictions.h5ad"), split["test_h5ad"],
                                              split["train_h5ad"], split["perturbation_key"])
                (logs_dir / "test_scores.json").write_text(json.dumps(job.score, indent=2), encoding="utf-8")
        else:
            script = logs_dir / "job.sbatch"
            script.write_text(render_sbatch(f"cf_{dataset_name}_{task_id}", command, logs_dir, slurm or {}, env))
            proc = subprocess.run(["sbatch", str(script)], text=True, capture_output=True)
            match = re.search(r"Submitted batch job (\d+)", proc.stdout)
            job = JobRecord(task_id, code_file, "submitted" if proc.returncode == 0 else "submit_failed", command,
                            str(logs_dir), job_id=match.group(1) if match else "", error=proc.stderr.strip())
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
