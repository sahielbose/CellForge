"""The expert pool and team selection for Method Design."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..llm import LLMInterface


@dataclass(frozen=True)
class Expert:
    name: str
    domain: str
    focus: str
    skills: tuple


EXPERTS = [
    Expert("Data Engineer", "data_engineering", "preprocessing pipelines for sparse, high-dimensional single-cell data",
           ("quality control", "normalization", "batch correction", "feature selection", "dimensionality reduction")),
    Expert("Single Cell Biology Expert", "single_cell_biology", "how cells respond to perturbations and how to read those responses",
           ("cell type annotation", "differential expression", "pathway enrichment", "cell state transitions")),
    Expert("Single Cell Model Architect", "single_cell_biology", "models of perturbation response and cell state dynamics",
           ("perturbation response modeling", "regulatory network inference", "multi-modal integration", "temporal dynamics")),
    Expert("Deep Learning Expert", "deep_learning", "neural architectures for perturbation prediction",
           ("architecture design", "attention mechanisms", "graph neural networks", "transfer learning")),
    Expert("Molecular Biology Specialist", "molecular_biology", "transcriptional regulation and signal transduction",
           ("pathway mapping", "molecular mechanisms", "epigenetic regulation", "protein-DNA interactions")),
    Expert("Bioinformatician", "bioinformatics", "network inference and multi-omics integration",
           ("statistical modeling", "gene regulatory networks", "algorithm development", "data integration")),
    Expert("Statistician", "statistics", "testing, effect sizes and experimental power in high-dimensional data",
           ("multiple testing correction", "effect size estimation", "power analysis", "mixed effects models")),
    Expert("Drug Discovery Expert", "drug_discovery", "small molecules and their targets",
           ("drug-target prediction", "structure-activity relationships", "molecular representations", "toxicity")),
    Expert("Gene Regulation Expert", "gene_regulation", "transcription factors and regulatory networks",
           ("TF binding analysis", "enhancer prediction", "regulatory network inference")),
    Expert("Experimental Designer", "experimental_design", "perturbation experiments, controls and replicates",
           ("control selection", "replicate planning", "time series design")),
    Expert("Self-critic", "validation", "finding flaws in proposed methods and their evaluation",
           ("model evaluation", "statistical validation", "method comparison", "leakage detection")),
    Expert("Baseline Assessment Expert", "baseline_assessment", "benchmarking against simple and published baselines",
           ("method benchmarking", "limitation analysis", "comparative evaluation")),
    Expert("Training Expert", "training", "training strategy and optimization",
           ("hyperparameter search", "regularization", "training monitoring", "compute budgeting")),
    Expert("Drug Response Expert", "drug_response", "dose-response and mechanism of action at single-cell resolution",
           ("dose-response modeling", "drug combinations", "mechanism of action")),
    Expert("Pathway Analyst", "pathway", "pathway and network structure of perturbation effects",
           ("gene set enrichment", "network modeling", "cell type-specific pathways")),
    Expert("Cell Communication Expert", "cell_communication", "ligand-receptor signaling between cells",
           ("ligand-receptor inference", "signaling networks", "cytokine responses")),
    Expert("Omics Modality Expert", "omics", "RNA, protein and chromatin modalities and their integration",
           ("CITE-seq", "scATAC-seq", "multi-omics integration")),
]

# The team always covers data, biology, modeling and training; the critic is separate.
CORE = ["Data Engineer", "Single Cell Biology Expert", "Deep Learning Expert", "Training Expert"]
CRITIC = "Self-critic"


class ExpertPool:
    def __init__(self, llm: Optional[LLMInterface] = None):
        self.experts = {expert.name: expert for expert in EXPERTS}
        self.llm = llm

    def get(self, name: str) -> Optional[Expert]:
        cleaned = re.sub(r"^[\s\-*\d.)]+|\*+", "", name).strip().lower()
        return next((e for key, e in self.experts.items() if key.lower() == cleaned), None)

    def select(self, task_type: str, task_analysis: Dict[str, Any], size: int = 4) -> List[Expert]:
        """Pick the discussion team: the core four plus LLM-chosen specialists, capped at `size`."""

        team = [self.experts[name] for name in CORE]
        if size > len(team) and self.llm is not None:
            try:
                extra = self._ask_llm(task_type, task_analysis)
            except Exception as exc:
                print(f"Expert selection fell back to the core team: {exc}")
                extra = []
            team += [e for e in extra if e not in team and e.name != CRITIC]
        return team[:size]

    def _ask_llm(self, task_type: str, task_analysis: Dict[str, Any]) -> List[Expert]:
        roster = "\n".join(f"- {e.name}: {e.focus}" for e in EXPERTS if e.name not in CORE + [CRITIC])
        prompt = (
            f"A {task_type} prediction task needs specialists beyond data engineering, single-cell biology, "
            f"deep learning and training.\n\nTask summary:\n{str(task_analysis)[:1500]}\n\nAvailable specialists:\n"
            f"{roster}\n\nList the two or three most useful specialists, one name per line, nothing else."
        )
        reply = self.llm.chat_completion([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=100)
        return [expert for line in reply.splitlines() if (expert := self.get(line))]
