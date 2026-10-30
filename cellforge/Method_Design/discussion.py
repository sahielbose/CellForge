"""Graph-structured expert discussion with critic and peer review.

Each round every expert drafts or revises a proposal, a central critic scores
every proposal, and each expert scores the others. Coordination scores follow
the paper:

    c_t = 0.3 * c_{t-1} + 0.4 * critic_t + 0.3 * mean(peer_t)

The discussion stops once every expert has c >= 0.8 and the spread of scores is
below 0.03 (after a minimum number of rounds), or when the round cap is hit. A
final call turns the transcript into the research plan.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm import parse_json
from .experts import Expert

PLAN_SCHEMA = {
    "summary": "",
    "model_architecture": {
        "name": "",
        "family": "",
        "components": [{"name": "", "purpose": "", "config": {}}],
        "rationale": "",
    },
    "data_processing": {"steps": [""], "split": ""},
    "training_strategy": {"objective": "", "optimizer": "", "schedule": "", "regularization": "", "early_stopping": ""},
    "evaluation_metrics": {"primary": [""], "baselines": [""], "de_genes": ""},
    "implementation_details": {"inputs": "", "outputs": "", "compute_budget": ""},
    "risks": [""],
    "evidence_ids": [""],
}


def clamp(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class DiscussionState:
    proposals: Dict[str, str] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    feedback: Dict[str, List[str]] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    consensus_reached: bool = False


class GraphDiscussion:
    def __init__(
        self,
        llm,
        experts: List[Expert],
        task_summary: str,
        evidence: str = "",
        max_rounds: int = 6,
        min_rounds: int = 3,
        threshold: float = 0.8,
        max_spread: float = 0.03,
        max_tokens: int = 900,
    ):
        self.llm = llm
        self.experts = experts
        self.task_summary = task_summary
        self.evidence = evidence
        self.max_rounds = max_rounds
        self.min_rounds = min(min_rounds, max_rounds)
        self.threshold = threshold
        self.max_spread = max_spread
        self.max_tokens = max_tokens
        self.state = DiscussionState(scores={e.name: 0.5 for e in experts})

    def _ask(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        return self.llm.chat_completion(
            [{"role": "user", "content": prompt}], max_tokens=max_tokens or self.max_tokens
        )

    def _ask_json(self, prompt: str) -> Dict[str, Any]:
        try:
            return parse_json(self._ask(prompt, max_tokens=600))
        except ValueError:
            return {}

    def _proposal_prompt(self, expert: Expert, round_index: int) -> str:
        previous = self.state.proposals.get(expert.name)
        notes = "\n".join(f"- {line}" for line in self.state.feedback.get(expert.name, [])) or "- none yet"
        others = "\n\n".join(
            f"{name}:\n{text[:1200]}" for name, text in self.state.proposals.items() if name != expert.name
        )
        return (
            f"You are the {expert.name}, focused on {expert.focus} "
            f"(skills: {', '.join(expert.skills)}).\n\nTASK\n{self.task_summary}\n\n"
            f"LITERATURE (cite evidence IDs in brackets)\n{self.evidence or 'none retrieved'}\n\n"
            + (f"YOUR PREVIOUS PROPOSAL\n{previous}\n\nREVIEWS OF IT\n{notes}\n\n" if previous else "")
            + (f"OTHER PROPOSALS\n{others}\n\n" if others else "")
            + f"Round {round_index + 1}. Propose the model design for this task from your angle: concrete "
            "architecture components and their configuration, how it handles unseen perturbations, the loss, "
            "and how it will be shown to beat the mean-shift baseline. Address every review point. "
            "Be specific and keep it under 400 words."
        )

    def _critic_prompt(self) -> str:
        proposals = "\n\n".join(f"### {name}\n{text}" for name, text in self.state.proposals.items())
        return (
            f"You are the area chair reviewing proposals for this task:\n{self.task_summary}\n\n{proposals}\n\n"
            "Score each proposal from 0 to 1 for scientific soundness, feasibility on one GPU, and whether its "
            'evaluation is leak-free. Return JSON: {"scores": {"<expert name>": 0.0}, '
            '"comments": {"<expert name>": "<the most important fix>"}}'
        )

    def _peer_prompt(self, reviewer: Expert) -> str:
        proposals = "\n\n".join(
            f"### {name}\n{text}" for name, text in self.state.proposals.items() if name != reviewer.name
        )
        return (
            f"You are the {reviewer.name}. Review these proposals from the standpoint of {reviewer.focus}.\n\n"
            f"{proposals}\n\nReturn JSON: "
            '{"scores": {"<expert name>": 0.0}, "comments": {"<expert name>": "<one concrete issue>"}}'
        )

    def run_round(self, round_index: int) -> Dict[str, Any]:
        for expert in self.experts:
            self.state.proposals[expert.name] = self._ask(self._proposal_prompt(expert, round_index))

        critic = self._ask_json(self._critic_prompt())
        critic_scores = {name: clamp(critic.get("scores", {}).get(name)) for name in self.state.proposals}
        feedback = {name: [f"Critic: {c}"] for name, c in critic.get("comments", {}).items() if name in critic_scores}
        peer_scores: Dict[str, List[float]] = {name: [] for name in self.state.proposals}
        for reviewer in self.experts:
            review = self._ask_json(self._peer_prompt(reviewer))
            for name, score in review.get("scores", {}).items():
                if name in peer_scores and name != reviewer.name:
                    peer_scores[name].append(clamp(score))
            for name, comment in review.get("comments", {}).items():
                if name in peer_scores and name != reviewer.name:
                    feedback.setdefault(name, []).append(f"{reviewer.name}: {comment}")

        for name in self.state.scores:
            peers = peer_scores.get(name) or [0.5]
            self.state.scores[name] = (
                0.3 * self.state.scores[name] + 0.4 * critic_scores.get(name, 0.5) + 0.3 * sum(peers) / len(peers)
            )
        self.state.feedback = feedback
        record = {
            "round": round_index + 1,
            "proposals": dict(self.state.proposals),
            "critic": critic,
            "peer_scores": peer_scores,
            "coordination_scores": dict(self.state.scores),
        }
        self.state.history.append(record)
        return record

    def converged(self, round_index: int) -> bool:
        scores = list(self.state.scores.values())
        return (
            round_index + 1 >= self.min_rounds
            and min(scores) >= self.threshold
            and max(scores) - min(scores) < self.max_spread
        )

    def run(self) -> DiscussionState:
        for round_index in range(self.max_rounds):
            record = self.run_round(round_index)
            scores = ", ".join(f"{k} {v:.2f}" for k, v in record["coordination_scores"].items())
            print(f"Round {round_index + 1}: {scores}")
            if self.converged(round_index):
                self.state.consensus_reached = True
                break
        return self.state

    def synthesize(self) -> Optional[Dict[str, Any]]:
        """Merge the final proposals into one plan, weighting experts by coordination score."""

        ranked = sorted(self.state.proposals, key=lambda n: self.state.scores.get(n, 0), reverse=True)
        proposals = "\n\n".join(
            f"### {name} (coordination score {self.state.scores.get(name, 0):.2f})\n{self.state.proposals[name]}"
            for name in ranked
        )
        prompt = (
            f"Merge these expert proposals into one research plan for the task below. Prefer the "
            f"higher-scored proposals where they conflict, keep only components that are justified, and "
            f"cite evidence IDs.\n\nTASK\n{self.task_summary}\n\nPROPOSALS\n{proposals}\n\n"
            f"Return only JSON with this structure:\n{json.dumps(PLAN_SCHEMA, indent=2)}"
        )
        try:
            plan = parse_json(self._ask(prompt, max_tokens=3000))
        except Exception as exc:
            print(f"Plan synthesis failed: {exc}")
            return None
        return plan if isinstance(plan.get("model_architecture"), dict) else None
