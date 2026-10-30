#!/usr/bin/env python3
"""CellForge command line: task analysis, method design, code generation, autorun."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from cellforge.paths import config_path as resolve_config_path
from cellforge.paths import data_path, resolve_workspace_path, workspace_root

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None and (workspace_root() / ".env").exists():
    load_dotenv(workspace_root() / ".env")

DEFAULT_TASK_DESCRIPTION = """Develop a model that predicts the single-cell gene expression profile produced by each perturbation in this dataset.

Input: the expression profile of unperturbed control cells and the identity of the perturbation.
Output: the predicted expression profile after perturbation.

Evaluation: split perturbation identities into five folds and test on held-out perturbations only, so the model is scored on perturbations it never saw in training. Report MSE, PCC and R2 over all genes and over the top 50 differentially expressed genes (MSE_DE, PCC_DE, R2_DE), DEG recall, and PCC on the change from control, and compare against the no-change and mean-shift baselines."""

LLM_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "LLAMA_API_KEY", "QWEN_API_KEY")
PHASES = ("task_analysis", "method_design", "code_generation")


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", (name or "").strip()).strip("_").lower() or "unknown_dataset"


def _dataset_name(dataset_path: str) -> str:
    path = Path(dataset_path or "")
    return _slug(path.stem if path.suffix else path.name)


def _latest(root: Path, pattern: str) -> Optional[Path]:
    files = list(root.rglob(pattern)) if root.exists() else []
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def approve(question: str) -> bool:
    """The paper's human checkpoints: a yes from the terminal, and no when nobody is there to answer."""

    if not sys.stdin.isatty():
        print(f"{question} -> needs an interactive terminal; stopping")
        return False
    return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}


def llm_configured() -> bool:
    return any(os.getenv(key) for key in LLM_KEYS) or bool(os.getenv("CUSTOM_API_URL") and os.getenv("CUSTOM_API_KEY"))


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    target = resolve_config_path(config_path)
    if target.exists():
        config = json.loads(target.read_text(encoding="utf-8"))
    else:
        config = {
            "task_description": DEFAULT_TASK_DESCRIPTION,
            "dataset_path": "data/datasets/",
            "workflow_phases": list(PHASES),
            "code_generation": {"backend": os.getenv("CODEGEN_BACKEND", "codex")},
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Created default configuration: {target}")
    config.setdefault("task_description", DEFAULT_TASK_DESCRIPTION)
    config.setdefault("workflow_phases", list(PHASES))
    return config


def run_task_analysis(config: Dict[str, Any]) -> bool:
    from cellforge.dataset import describe_dataset
    from cellforge.retrieval import LiteratureRetriever
    from cellforge.Task_Analysis.main import run_task_analysis as analyze

    print("\n== Phase 1: task analysis ==")
    dataset_path = config.get("dataset_path", "")
    try:
        dataset_info = describe_dataset(str(resolve_workspace_path(dataset_path)))
    except Exception as exc:
        print(f"Could not read the dataset ({exc}); continuing with its name only")
        dataset_info = {"dataset_name": _dataset_name(dataset_path), "dataset_path": dataset_path}
    out = data_path("analyses", _slug(dataset_info["dataset_name"]))
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TASK_ANALYSIS_OUTPUT_DIR"] = str(out)
    try:
        retriever = LiteratureRetriever.from_env(trace_dir=out / "retrieval")
        return analyze(config["task_description"], dataset_info, retriever=retriever) is not None
    except Exception as exc:
        print(f"Task analysis failed: {exc}")
        return False


def run_method_design(config: Dict[str, Any]) -> bool:
    from cellforge.Method_Design import generate_research_plan
    from cellforge.Method_Design.main import load_task_analysis
    from cellforge.retrieval import LiteratureRetriever

    print("\n== Phase 2: method design ==")
    dataset_path = config.get("dataset_path")
    root = data_path("analyses", _dataset_name(dataset_path)) if dataset_path else data_path("analyses")
    report = _latest(root, "task_analysis_*.json") or _latest(data_path("analyses"), "task_analysis_*.json")
    if report is None:
        print("No task analysis report found; run the task_analysis phase first")
        return False
    try:
        task_analysis = load_task_analysis(file_path=str(report), latest=False)
        out = data_path("plans", _slug(task_analysis.get("dataset", {}).get("name", "unknown_dataset")))
        retriever = LiteratureRetriever.from_env(trace_dir=out / "retrieval")
        plan = generate_research_plan(task_analysis=task_analysis, rag_retriever=retriever, output_dir=str(out))
    except Exception as exc:
        print(f"Method design failed: {exc}")
        return False
    files = plan.get("generated_files", {})
    print(f"Research plan written to {files.get('output_directory', out)}")
    return bool(plan)


def run_code_generation(config: Dict[str, Any]) -> bool:
    from cellforge.autorun.runner import code_for, plan_task_ids

    print("\n== Phase 3: code generation ==")
    dataset_path = config.get("dataset_path")
    root = data_path("plans", _dataset_name(dataset_path)) if dataset_path else data_path("plans")
    plan_file = _latest(root, "research_plan_*.json") or _latest(data_path("plans"), "research_plan_*.json")
    if plan_file is None:
        print("No research plan found; run the method_design phase first")
        return False
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if config.get("review"):
        architecture = plan.get("model_architecture", {}).get("name", "unnamed")
        print(f"Plan {plan_file}\n  source: {plan.get('plan_source')}\n  model: {architecture}\n  {plan.get('summary', '')}")
        if not approve("Generate code for this plan?"):
            return False
    backend = config.get("code_generation", {}).get("backend", "codex")
    task_id = os.getenv("CODEGEN_TASK_ID", "").strip()
    tasks = [task_id] if task_id else plan_task_ids(plan)
    written = 0
    for task in tasks:
        try:
            code = code_for(plan, plan_file, task, backend)
        except Exception as exc:
            print(f"{task}: code generation failed: {exc}")
            continue
        if code and Path(code).exists():
            written += 1
            print(f"{task}: verified code written to {code}")
        else:
            print(f"{task}: no verified script")
    return written == len(tasks)


def run_autorun_phase(config: Dict[str, Any], args: argparse.Namespace) -> bool:
    from cellforge.autorun import run_autorun

    print("\n== Phase 4: autorun ==")
    slurm = {"partition": args.partition, "time": args.slurm_time, "cpus": args.cpus_per_task,
             "mem": args.mem, "gres": args.gres, "conda_env": args.conda_env}
    try:
        summary = run_autorun(
            dataset_path=config["dataset_path"],
            plans_dir=args.autorun_plan_dir,
            max_tasks=args.max_tasks,
            executor=args.executor,
            slurm=slurm,
            n_folds=args.folds,
            fold=args.fold,
            split_val_ratio=args.split_val_ratio,
            split_seed=args.split_seed,
            codegen_backend=args.codegen_backend,
            approve=(lambda code, command: approve(f"Run {code}\n  {command}\n?")) if args.review else None,
        )
    except Exception as exc:
        print(f"Autorun failed: {exc}")
        return False
    for job in summary["jobs"]:
        score = (job.get("score") or {}).get("metrics", {})
        line = f"{job['task_id']}: {job['status']}"
        if score.get("model"):
            line += f"  PCC_delta {score['model']['PCC_delta']:.3f} (mean-shift {score['mean_shift']['PCC_delta']:.3f})"
        print(line)
    if args.executor == "slurm":
        print(f"After the jobs finish: python -m cellforge.autorun.scoring {summary['run_root']}")
    return all(job["status"] in {"completed", "submitted"} for job in summary["jobs"])


def run_complete_workflow(config: Dict[str, Any]) -> bool:
    if not llm_configured():
        print("No LLM provider key is set; copy .env.example to .env and add one")
        return False
    runners = {"task_analysis": run_task_analysis, "method_design": run_method_design,
               "code_generation": run_code_generation}
    for phase in config["workflow_phases"]:
        if phase in runners and not runners[phase](config):
            print(f"Stopped after {phase} failed")
            return False
    print(f"\nAll phases finished. Results are under {data_path()}")
    return True


def init_workspace() -> None:
    for name in ("datasets", "analyses", "plans", "codes", "discussion"):
        data_path(name).mkdir(parents=True, exist_ok=True)
    print(f"Created data/ folders under {workspace_root()}")


def run_doctor(config_name: str = "config.json") -> bool:
    literature = Path(os.getenv("CELLFORGE_LITERATURE_DIR", str(data_path("literature")))).expanduser()
    checks = [
        ("configuration", resolve_config_path(config_name).exists(), True),
        ("datasets directory", data_path("datasets").is_dir(), True),
        ("literature directory", literature.is_dir(), False),
        ("LLM provider key", llm_configured(), False),
        ("Python 3.9 or newer", sys.version_info >= (3, 9), True),
    ]
    print(f"Workspace: {workspace_root()}")
    for name, ok, _ in checks:
        print(f"  {'ok  ' if ok else 'MISSING'} {name}")
    return all(ok for _, ok, required in checks if required)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CellForge: agentic design of virtual cell models")
    parser.add_argument("--config", default="config.json", help="configuration file")
    parser.add_argument("--workspace", help="runtime workspace root (or set CELLFORGE_WORKSPACE_DIR)")
    parser.add_argument("--init", action="store_true", help="create the data/ folders and a default config")
    parser.add_argument("--doctor", action="store_true", help="check the workspace and exit")
    parser.add_argument("--phase", choices=[*PHASES, "autorun"], help="run one phase only")
    parser.add_argument("--dataset-path", help=".h5ad file or a folder containing one")
    parser.add_argument("--task", help="task description text")
    parser.add_argument("--task-file", help="file containing the task description")
    parser.add_argument("--review", action="store_true",
                        help="ask before code generation and before each training job, as in the paper")
    parser.add_argument("--codegen-backend", default=os.getenv("CODEGEN_BACKEND", "codex"), choices=["codex"])
    run = parser.add_argument_group("autorun")
    run.add_argument("--executor", choices=["local", "slurm"], default="local")
    run.add_argument("--max-tasks", type=int, default=None, help="limit the number of task-wise jobs")
    run.add_argument("--autorun-plan-dir", default=None, help="plan folder override")
    run.add_argument("--folds", type=int, default=5, help="number of perturbation folds, as in the paper")
    run.add_argument("--fold", type=int, default=0, help="which fold to hold out for test")
    run.add_argument("--split-val-ratio", type=float, default=0.1, help="share of perturbations held out for validation")
    run.add_argument("--split-seed", type=int, default=42)
    run.add_argument("--partition", default="gpu")
    run.add_argument("--slurm-time", default="08:00:00")
    run.add_argument("--cpus-per-task", type=int, default=4)
    run.add_argument("--mem", default="32G")
    run.add_argument("--gres", default="gpu:1")
    run.add_argument("--conda-env", default="", help="conda environment to activate in Slurm jobs")
    return parser


def main() -> bool:
    args = build_parser().parse_args()
    if args.workspace:
        os.environ["CELLFORGE_WORKSPACE_DIR"] = str(Path(args.workspace).expanduser().resolve())
        if load_dotenv is not None and (workspace_root() / ".env").exists():
            load_dotenv(workspace_root() / ".env", override=False)
    if args.doctor:
        return run_doctor(args.config)
    if args.init:
        init_workspace()
        load_config(args.config)
        return True

    config = load_config(args.config)
    if args.dataset_path:
        config["dataset_path"] = args.dataset_path
    if args.task_file:
        config["task_description"] = resolve_workspace_path(args.task_file).read_text(encoding="utf-8").strip()
    elif args.task:
        config["task_description"] = args.task.strip()
    config.setdefault("code_generation", {})["backend"] = args.codegen_backend
    config["review"] = args.review

    if args.phase == "autorun":
        return run_autorun_phase(config, args)
    if args.phase:
        return {"task_analysis": run_task_analysis, "method_design": run_method_design,
                "code_generation": run_code_generation}[args.phase](config)
    return run_complete_workflow(config)


def cli_entrypoint() -> int:
    return 0 if main() else 1


if __name__ == "__main__":
    raise SystemExit(cli_entrypoint())
