"""Research plan helpers: the fallback plan, Markdown, Mermaid, and task-wise blocks."""

from typing import Any, Dict, List

from ..Task_Analysis.report import render_markdown


def fallback_plan(task_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """A conservative plan used only when synthesis fails, labeled as such.

    It starts from the Task Analysis recommendations when they exist, and from a
    mean-shift baseline plus a small conditional MLP otherwise.
    """

    recs = task_analysis.get("analysis", {}).get("final_recommendations", {}) or {}
    return {
        "plan_source": "template_fallback",
        "summary": "Fallback plan: predict the change from control with a perturbation-conditioned MLP "
        "and report it against the mean-shift baseline.",
        "model_architecture": recs.get("model_architecture") or {
            "name": "ConditionalMLP",
            "family": "perturbation-conditioned regression on the change from control",
            "components": [
                {"name": "perturbation embedding", "purpose": "map each target to a learned vector", "config": {"dim": 64}},
                {"name": "MLP decoder", "purpose": "predict the expression shift", "config": {"hidden": [512, 512]}},
            ],
            "rationale": "Small enough to train in minutes, and directly comparable to the mean-shift baseline.",
        },
        "data_processing": {"steps": recs.get("data_processing") or ["normalize to 1e4 counts", "log1p", "top 2,000 HVGs"],
                            "split": "hold out perturbations, never cells, for testing"},
        "training_strategy": recs.get("training_strategy") or {"objective": "MSE on the change from control",
                                                               "optimizer": "AdamW, lr 1e-3", "early_stopping": "on validation loss"},
        "evaluation_metrics": recs.get("evaluation_protocol") or {"primary": ["MSE", "PCC", "PCC on delta", "MSE_DE"],
                                                                  "baselines": ["no change", "mean shift"]},
        "implementation_details": {},
        "risks": ["This plan was not produced by the expert discussion."],
        "evidence_ids": [],
    }


def task_wise_plan(task_analysis: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split the plan into code-generation tasks, one per objective."""

    blocks = []
    for index, objective in enumerate(task_analysis.get("objectives") or ["Predict post-perturbation expression"], 1):
        blocks.append({
            "task_id": f"T{index:02d}",
            "objective": objective,
            "scope": {"task_type": task_analysis.get("task_type"), "dataset": task_analysis.get("dataset", {})},
            "implementation_contract": {
                "inputs": ["control expression", "perturbation identity"],
                "outputs": ["predicted expression per held-out perturbation", "validation metrics JSON"],
                "evaluation_metrics": task_analysis.get("evaluation_metrics", []),
            },
            "design_refs": {key: plan.get(key, {}) for key in
                            ("model_architecture", "data_processing", "training_strategy", "implementation_details")},
        })
    return blocks


def plan_markdown(plan: Dict[str, Any], discussion: Dict[str, Any]) -> str:
    rounds = discussion.get("rounds", 0)
    status = "reached" if discussion.get("consensus_reached") else "not reached"
    body = {k: v for k, v in plan.items() if k not in {"task_wise_plan", "summary"}}
    return (
        f"# Research plan\n\n{plan.get('summary', '')}\n\n"
        f"Plan source: {plan.get('plan_source', 'expert_discussion')}. "
        f"Discussion ran {rounds} round(s); consensus {status}.\n\n{render_markdown(body, level=2)}\n"
    )


def plan_mermaid(plan: Dict[str, Any]) -> str:
    """Draw the data flow through the planned architecture components."""

    components = plan.get("model_architecture", {}).get("components") or []
    lines = ["flowchart LR", '    data["control expression + perturbation"]']
    previous = "data"
    for index, component in enumerate(components):
        node = f"c{index}"
        label = str(component.get("name", f"component {index + 1}")).replace('"', "'")
        lines.append(f'    {node}["{label}"]')
        lines.append(f"    {previous} --> {node}")
        previous = node
    lines += ['    out["predicted expression"]', f"    {previous} --> out"]
    return "\n".join(lines)
