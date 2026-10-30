"""Method Design: an expert discussion that converges on a research plan."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from ..llm import USAGE, LLMInterface, usage_since
from ..paths import data_path, resolve_workspace_path
from ..Task_Analysis.agents import format_evidence
from .discussion import PLAN_SCHEMA, GraphDiscussion
from .experts import ExpertPool
from .main import normalize_task_analysis, task_summary
from .plan import fallback_plan, plan_markdown, plan_mermaid, task_wise_plan


def discussion_evidence(retriever, task_analysis: Dict[str, Any], top_k: int = 4) -> str:
    """Retrieve modeling literature once for the whole discussion."""

    if retriever is None:
        return ""
    dataset = task_analysis.get("dataset", {}).get("name", "")
    task_type = str(task_analysis.get("task_type", "")).replace("_", " ")
    papers = []
    for query in (f"{task_type} response prediction model", f"{dataset} perturbation prediction benchmark"):
        try:
            papers += retriever.retrieve(query.strip(), top_k=top_k)
        except Exception as exc:
            print(f"Retrieval failed for '{query}': {exc}")
    return format_evidence(papers, limit=2 * top_k)


def generate_research_plan(
    task_analysis: Union[str, Dict[str, Any]],
    rag_retriever=None,
    output_dir: str = "data/plans",
    llm=None,
    **_ignored,
) -> Dict[str, Any]:
    if isinstance(task_analysis, str):
        task_analysis = json.loads(Path(task_analysis).read_text(encoding="utf-8"))
    start = dict(USAGE)
    task_analysis = normalize_task_analysis(task_analysis)
    llm = llm or LLMInterface()

    size = int(os.getenv("METHOD_DESIGN_MAX_EXPERTS", "4"))
    experts = ExpertPool(llm).select(task_analysis["task_type"], task_analysis, size=size)
    print("Experts: " + ", ".join(e.name for e in experts))
    discussion = GraphDiscussion(
        llm,
        experts,
        task_summary(task_analysis),
        evidence=discussion_evidence(rag_retriever, task_analysis),
        max_rounds=min(20, int(os.getenv("METHOD_DESIGN_MAX_ROUNDS", "6"))),
        min_rounds=int(os.getenv("METHOD_DESIGN_MIN_ROUNDS", "1")),
        reviews=tuple(r.strip() for r in os.getenv("METHOD_DESIGN_REVIEWS", "critic,peer").split(",") if r.strip()),
        max_tokens=int(os.getenv("METHOD_DESIGN_MAX_TOKENS_PER_CALL", "900")),
    )
    state = discussion.run()
    plan = discussion.synthesize() or fallback_plan(task_analysis)
    plan.setdefault("plan_source", "expert_discussion")
    plan["task_wise_plan"] = task_wise_plan(task_analysis, plan)

    summary = {
        "rounds": len(state.history),
        "consensus_reached": state.consensus_reached,
        "coordination_scores": state.scores,
        "experts": [e.name for e in experts],
        "llm_usage": usage_since(start),
    }
    trace = getattr(rag_retriever, "last_trace", None)
    result = {
        "json": plan,
        "markdown": plan_markdown(plan, summary),
        "mermaid": plan_mermaid(plan),
        "discussion_summary": summary,
        "retrieval": trace.to_dict() if trace else None,
    }
    result["generated_files"] = save_plan(result, state.history, output_dir, task_analysis)
    return result


def save_plan(result: Dict[str, Any], history, output_dir: str, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
    out = resolve_workspace_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = f"research_plan_{datetime.now():%Y%m%d_%H%M%S}"
    (out / f"{base}.json").write_text(json.dumps(result["json"], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"{base}.md").write_text(result["markdown"], encoding="utf-8")
    (out / f"{base}.mmd").write_text(result["mermaid"], encoding="utf-8")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", task_analysis["dataset"]["name"]).strip("_").lower() or "unknown_dataset"
    log_dir = data_path("discussion", slug)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{base}_discussion.json").write_text(
        json.dumps({"summary": result["discussion_summary"], "rounds": history}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"base_filename": base, "output_directory": str(out), "files": [f"{base}.{ext}" for ext in ("json", "md", "mmd")]}


__all__ = ["PLAN_SCHEMA", "GraphDiscussion", "ExpertPool", "generate_research_plan"]
