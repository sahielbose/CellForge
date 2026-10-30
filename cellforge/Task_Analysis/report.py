"""Data structures for the Task Analysis report."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class AnalysisResult:
    content: Dict[str, Any]
    agent: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "agent": self.agent,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class TaskAnalysisReport:
    dataset_analysis: AnalysisResult
    problem_investigation: AnalysisResult
    baseline_assessment: AnalysisResult
    refinement_comments: List[Dict[str, Any]]
    final_recommendations: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_analysis": self.dataset_analysis.to_dict(),
            "problem_investigation": self.problem_investigation.to_dict(),
            "baseline_assessment": self.baseline_assessment.to_dict(),
            "refinement_comments": self.refinement_comments,
            "final_recommendations": self.final_recommendations,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_markdown(self) -> str:
        sections = [
            ("Dataset analysis", self.dataset_analysis.content),
            ("Problem investigation", self.problem_investigation.content),
            ("Baseline assessment", self.baseline_assessment.content),
            ("Final recommendations", self.final_recommendations),
        ]
        lines = ["# Task analysis report", f"Generated {self.timestamp:%Y-%m-%d %H:%M}", ""]
        for title, content in sections:
            lines += [f"## {title}", render_markdown(content, level=3), ""]
        for index, comment in enumerate(self.refinement_comments, start=1):
            lines += [f"## Refinement round {index}", render_markdown(comment, level=3), ""]
        return "\n".join(lines)


def render_markdown(value: Any, level: int = 3) -> str:
    """Render nested JSON as headings and bullets, whatever keys the model returned."""

    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            name = str(key).replace("_", " ").capitalize()
            if isinstance(item, (dict, list)):
                out += [f"{'#' * min(level, 6)} {name}", render_markdown(item, level + 1)]
            else:
                out.append(f"- {name}: {item}")
        return "\n".join(out)
    if isinstance(value, list):
        return "\n".join(f"- {render_markdown(item, level) if isinstance(item, dict) else item}" for item in value)
    return str(value)
