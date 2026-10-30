"""The four Task Analysis agents.

Three analysts each read the task, the real dataset summary and the retrieved
evidence, and return a JSON section. The Refinement Agent then reviews all three
sections together, rewrites the ones it has comments on, and writes the final
recommendations that Method Design starts from.
"""

import json
from typing import Any, Dict, List, Optional

from ..llm import LLMInterface, parse_json
from .report import AnalysisResult, TaskAnalysisReport

KNOWN_BASELINES = {
    "mean baseline": ["mean of training perturbations", "additive baseline", "linear baseline"],
    "linear model": ["linear regression", "linear model", "ridge"],
    "GEARS": ["gears"],
    "CPA": ["compositional perturbation autoencoder", " cpa "],
    "scGen": ["scgen"],
    "scGPT": ["scgpt"],
    "random forest": ["random forest"],
}


def format_evidence(papers: List[Dict[str, Any]], limit: int = 8, chars: int = 400) -> str:
    """Render retrieved papers with their evidence IDs so claims can cite them."""

    lines = []
    for paper in papers[:limit]:
        year = paper.get("metadata", {}).get("year") or ""
        text = " ".join(str(paper.get("content") or paper.get("abstract") or "").split())[:chars]
        lines.append(f"[{paper.get('evidence_id', 'untracked')}] {paper.get('title', 'Untitled')} {year}\n  {text}")
    return "\n".join(lines) or "No literature was retrieved."


def baselines_mentioned(papers: List[Dict[str, Any]]) -> List[str]:
    found = []
    for paper in papers:
        text = f" {paper.get('title', '')} {paper.get('content') or paper.get('abstract') or ''} ".lower()
        found += [name for name, keys in KNOWN_BASELINES.items() if name not in found and any(k in text for k in keys)]
    return found


class AnalysisAgent:
    name = "Analyst"
    system = "You are an expert in single-cell perturbation research. Reply with one valid JSON object."
    instructions = ""
    schema: Dict[str, Any] = {}

    def __init__(self, llm: Optional[LLMInterface] = None):
        self.llm = llm or LLMInterface()

    def _run_llm(self, prompt: str) -> Dict[str, Any]:
        content = self.llm.generate(prompt, self.system).get("content") or ""
        try:
            return parse_json(content)
        except ValueError:
            return {"content": content, "error": "Failed to parse JSON response"}

    def prompt(self, task: str, dataset: Dict[str, Any], evidence: List[Dict[str, Any]]) -> str:
        return (
            f"{self.instructions}\n\nTASK\n{task}\n\nDATASET (read from the file)\n"
            f"{json.dumps(dataset, indent=2, default=str)}\n\nLITERATURE (cite evidence IDs in brackets)\n"
            f"{format_evidence(evidence)}\n\nReturn JSON with this structure, filling every field "
            f"with specific content for this dataset:\n{json.dumps(self.schema, indent=2)}"
        )

    def analyze(self, task: str, dataset: Dict[str, Any], evidence: List[Dict[str, Any]]) -> AnalysisResult:
        content = self._run_llm(self.prompt(task, dataset, evidence))
        return AnalysisResult(content=content, agent=self.name, metadata={"evidence_ids": [p.get("evidence_id") for p in evidence]})


class DatasetAnalyst(AnalysisAgent):
    name = "Dataset Analyst"
    instructions = (
        "Assess this perturbation dataset: experimental design (perturbations, controls, replicates, "
        "batch structure), data characteristics (sparsity, cells per perturbation, technical and "
        "biological noise), preprocessing that suits this assay, and quality risks. Base numbers on the "
        "dataset summary above; say when a quantity is unknown instead of guessing."
    )
    schema = {
        "experimental_design": {"perturbations": "", "controls": "", "replicates_and_batches": ""},
        "data_characteristics": {"scale": "", "sparsity": "", "noise_sources": [""]},
        "preprocessing": {"quality_control": [""], "normalization": "", "feature_selection": "", "batch_correction": ""},
        "quality_assessment": {"risks": [""], "mitigations": [""]},
    }


class ProblemInvestigator(AnalysisAgent):
    name = "Problem Investigator"
    instructions = (
        "Define the prediction problem formally: inputs, outputs, and what generalization is being "
        "tested (unseen perturbations, unseen combinations, unseen cell contexts). Identify the "
        "biological and statistical challenges, the research questions worth answering, and the "
        "analysis methods that fit them."
    )
    schema = {
        "formal_definition": {"input": "", "output": "", "generalization_setting": ""},
        "key_challenges": [{"challenge": "", "why_it_matters": "", "evidence": ""}],
        "research_questions": [""],
        "analysis_methods": [{"method": "", "purpose": ""}],
    }


class BaselineAssessor(AnalysisAgent):
    name = "Baseline Assessor"
    instructions = (
        "Assess baselines for this task. Always include the trivial baselines (predict no change from "
        "control, and predict the mean shift of training perturbations), because published deep models "
        "often fail to beat them. Give the evaluation protocol: split by held-out perturbation, metrics "
        "on all genes and on the top differentially expressed genes, and metrics on the change from control."
    )
    schema = {
        "baseline_models": [{"name": "", "principle": "", "strengths": "", "limitations": ""}],
        "evaluation_framework": {"split": "", "metrics": [""], "de_gene_definition": ""},
        "performance_analysis": "",
        "improvement_suggestions": [""],
    }

    def prompt(self, task, dataset, evidence):
        found = baselines_mentioned(evidence)
        hint = f"\n\nBaselines named in the retrieved literature: {', '.join(found)}." if found else ""
        return super().prompt(task, dataset, evidence) + hint


class RefinementAgent(AnalysisAgent):
    name = "Refinement Agent"
    system = "You review and integrate scientific analyses. Reply with one valid JSON object."
    sections = ("dataset", "problem", "baseline")
    recommendation_schema = {
        "data_processing": [""],
        "model_architecture": {"family": "", "rationale": ""},
        "training_strategy": [""],
        "evaluation_protocol": {"split": "", "metrics": [""], "baselines": [""]},
        "implementation_roadmap": [""],
    }

    def __init__(self, llm: Optional[LLMInterface] = None, max_rounds: int = 2):
        super().__init__(llm)
        self.max_rounds = max_rounds

    def refine(
        self,
        task: str,
        dataset: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        analyses: List[AnalysisResult],
    ) -> TaskAnalysisReport:
        current = dict(zip(self.sections, analyses, strict=True))
        history = []
        for _ in range(self.max_rounds):
            review = self._run_llm(self._review_prompt(task, dataset, evidence, current))
            comments = {key: review.get(key) for key in self.sections if review.get(key)}
            history.append(comments)
            if not comments:  # nothing left to fix, or the review was unparseable
                break
            for key, comment in comments.items():
                revised = self._run_llm(self._revise_prompt(current[key], comment))
                if "error" not in revised:  # keep the previous version when a revision fails
                    current[key] = AnalysisResult(revised, current[key].agent, metadata=current[key].metadata)
        recommendations = self._run_llm(self._recommend_prompt(task, dataset, evidence, current))
        return TaskAnalysisReport(current["dataset"], current["problem"], current["baseline"], history, recommendations)

    def _review_prompt(self, task, dataset, evidence, current) -> str:
        sections = {key: result.content for key, result in current.items()}
        return (
            f"Review three analyses of the same task for errors, gaps and contradictions with each other, "
            f"the dataset summary, or the literature.\n\nTASK\n{task}\n\nDATASET\n{json.dumps(dataset, default=str)}"
            f"\n\nLITERATURE\n{format_evidence(evidence, limit=5)}\n\nANALYSES\n{json.dumps(sections, indent=2)}\n\n"
            'Return {"dataset": [...], "problem": [...], "baseline": [...]}, each a list of concrete fixes. '
            "Use an empty list for a section that needs no change."
        )

    @staticmethod
    def _revise_prompt(result: AnalysisResult, comments: Any) -> str:
        return (
            f"Revise this {result.agent} section to address the reviewer comments. Keep its JSON structure, "
            f"mark assumptions explicitly, and return only the revised JSON.\n\nSECTION\n"
            f"{json.dumps(result.content, indent=2)}\n\nCOMMENTS\n{json.dumps(comments, indent=2)}"
        )

    def _recommend_prompt(self, task, dataset, evidence, current) -> str:
        sections = {key: result.content for key, result in current.items()}
        return (
            f"Write the final recommendations a model designer should start from.\n\nTASK\n{task}\n\nDATASET\n"
            f"{json.dumps(dataset, default=str)}\n\nANALYSES\n{json.dumps(sections, indent=2)}\n\n"
            f"Return JSON with this structure:\n{json.dumps(self.recommendation_schema, indent=2)}"
        )
