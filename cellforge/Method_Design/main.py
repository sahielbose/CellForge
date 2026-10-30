"""Load a Task Analysis report for Method Design without discarding its content."""

import json
import re
from typing import Any, Dict, Optional

from ..dataset import infer_task_type
from ..paths import data_path, resolve_workspace_path

METRICS = ("MSE_DE", "PCC_DE", "R2_DE", "MSE", "PCC", "R2")


def normalize_task_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map a saved report onto the fields Method Design reads, keeping the full analysis."""

    if "analysis" in data and "dataset" in data:
        return data
    task = data.get("task_description", "")
    dataset_info = data.get("dataset_info") or {}
    analysis = data.get("analysis_results") or {}
    text = task.upper()
    return {
        "task_type": data.get("task_type") or infer_task_type(task, dataset_info),
        "task_description": task,
        "dataset": {"name": dataset_info.get("dataset_name", "unknown_dataset"), **dataset_info},
        "analysis": analysis,
        "objectives": ["Predict post-perturbation expression for held-out perturbations"],
        "evaluation_metrics": [m for m in METRICS if re.search(rf"\b{m}\b", text)] or ["MSE", "PCC", "MSE_DE", "PCC_DE"],
    }


def task_summary(task_analysis: Dict[str, Any], limit: int = 6000) -> str:
    """The text every expert sees: the task, the dataset facts and the analysis conclusions."""

    analysis = task_analysis.get("analysis", {})
    picks = {
        "dataset": task_analysis.get("dataset"),
        "final_recommendations": analysis.get("final_recommendations"),
        "problem": (analysis.get("problem_investigation") or {}).get("content"),
        "baselines": (analysis.get("baseline_assessment") or {}).get("content"),
    }
    text = f"{task_analysis.get('task_description', '')}\n\nTask type: {task_analysis.get('task_type')}\n"
    text += json.dumps({k: v for k, v in picks.items() if v}, indent=1, default=str)
    return text[:limit]


def load_task_analysis(file_path: Optional[str] = None, latest: bool = True) -> Dict[str, Any]:
    if not file_path:
        reports = list(data_path("analyses").rglob("task_analysis_*.json")) if latest else []
        if not reports:
            raise FileNotFoundError(f"No task analysis reports under {data_path('analyses')}")
        file_path = str(max(reports, key=lambda p: p.stat().st_mtime))
    with resolve_workspace_path(file_path).open(encoding="utf-8") as handle:
        return normalize_task_analysis(json.load(handle))
