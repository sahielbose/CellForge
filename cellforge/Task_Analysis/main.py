"""Run Task Analysis: retrieve evidence once, run the three analysts, refine, save."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..dataset import infer_task_type
from ..paths import data_path
from .agents import BaselineAssessor, DatasetAnalyst, ProblemInvestigator, RefinementAgent
from .report import TaskAnalysisReport


def gather_evidence(retriever, task: str, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Two queries cover the dataset's own literature and the evaluation literature."""

    papers = retriever.search_for_task(task, dataset, limit=10)
    papers += retriever.search_evaluation_frameworks(task)
    unique = {}
    for paper in papers:
        unique.setdefault(paper.get("evidence_id") or paper.get("title"), paper)
    return list(unique.values())


def run_task_analysis(
    task_description: str,
    dataset_info: Optional[Dict[str, Any]] = None,
    retriever=None,
    llm=None,
    output_dir: Optional[str] = None,
) -> TaskAnalysisReport:
    from ..retrieval import LiteratureRetriever

    dataset_info = dataset_info or {}
    retriever = retriever or LiteratureRetriever.from_env()
    evidence = gather_evidence(retriever, task_description, dataset_info)
    print(f"Retrieved {len(evidence)} papers")

    analyses = []
    for agent in (DatasetAnalyst(llm), ProblemInvestigator(llm), BaselineAssessor(llm)):
        print(f"Running {agent.name}")
        analyses.append(agent.analyze(task_description, dataset_info, evidence))

    rounds = int(os.getenv("TASK_ANALYSIS_MAX_REFINEMENT_ROUNDS", "2"))
    report = RefinementAgent(llm, max_rounds=rounds).refine(task_description, dataset_info, evidence, analyses)
    save_report(report, task_description, dataset_info, retriever, output_dir)
    return report


def save_report(report, task_description, dataset_info, retriever, output_dir=None) -> Path:
    out = Path(output_dir or os.getenv("TASK_ANALYSIS_OUTPUT_DIR") or data_path("analyses", "unknown_dataset"))
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace = getattr(retriever, "last_trace", None)
    payload = {
        "timestamp": stamp,
        "task_description": task_description,
        "dataset_info": dataset_info,
        "task_type": infer_task_type(task_description, dataset_info),
        "analysis_results": report.to_dict(),
        "metadata": {"version": "2.0", "retrieval": trace.to_dict() if trace else None},
    }
    path = out / f"task_analysis_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    path.with_suffix(".md").write_text(report.to_markdown(), encoding="utf-8")
    print(f"Task analysis saved to {path}")
    return path
