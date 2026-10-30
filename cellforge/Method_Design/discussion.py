"""Graph-structured expert discussion with critic and peer review.

Each round every expert drafts or revises a proposal, a central critic scores
every proposal, and each expert scores the others. Experts draft from the
previous round's proposals only, so first drafts are independent and no expert
sees a draft written earlier in the same round. Reviewers see proposals under
neutral labels in a fresh random order, so neither the author's role nor the
position in the list can earn a higher score. Coordination scores follow the
paper:

    c_t = 0.3 * c_{t-1} + 0.4 * critic_t + 0.3 * mean(peer_t)

Scores lie in [0.1, 1.0] and c starts at 0.5. The discussion stops once every
expert has c >= 0.8 and the largest pairwise difference is below 0.03, or at the
round cap (the paper uses 20). The critic then turns the top-ranked proposal into
the research plan, and lower-ranked proposals are dropped.
"""

import json
import random
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
        return max(0.1, min(1.0, float(value)))
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
        min_rounds: int = 1,
        threshold: float = 0.8,
        max_spread: float = 0.03,
        max_tokens: int = 900,
        reviews: tuple = ("critic", "peer"),
        seed: int = 0,
    ):
        if not set(reviews) & {"critic", "peer"}:
            raise ValueError("The discussion needs critic or peer reviews to score proposals")
        self.reviews = set(reviews)
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
        self.rng = random.Random(seed)

    def _ask(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        return self.llm.chat_completion(
            [{"role": "user", "content": prompt}], max_tokens=max_tokens or self.max_tokens
        )

    def _ask_json(self, prompt: str) -> Dict[str, Any]:
        try:
            return parse_json(self._ask(prompt, max_tokens=600))
        except ValueError:
            return {}

    def _proposal_prompt(self, expert: Expert, round_index: int, previous_round: Dict[str, str]) -> str:
        previous = previous_round.get(expert.name)
        notes = "\n".join(f"- {line}" for line in self.state.feedback.get(expert.name, [])) or "- none yet"
        others = "\n\n".join(f"{name}:\n{text[:1200]}" for name, text in previous_round.items() if name != expert.name)
        return (
            f"You are the {expert.name}, focused on {expert.focus} "
            f"(skills: {', '.join(expert.skills)}).\n\nTASK\n{self.task_summary}\n\n"
            f"LITERATURE (cite evidence IDs in brackets)\n{self.evidence or 'none retrieved'}\n\n"
            + (f"YOUR PREVIOUS PROPOSAL\n{previous}\n\nREVIEWS OF IT\n{notes}\n\n" if previous else "")
            + (f"OTHER PROPOSALS FROM THE LAST ROUND\n{others}\n\n" if others else "")
            + f"Round {round_index + 1}. Propose the model design for this task from your angle: concrete "
            "architecture components and their configuration, how it handles unseen perturbations, the loss, "
            "and how it will be shown to beat the mean-shift baseline. Address every review point. "
            "Be specific and keep it under 400 words."
        )

    def _labels(self, names: List[str]) -> Dict[str, str]:
        """Neutral labels in a fresh random order, mapped back to the experts who wrote the proposals."""

        return {f"Proposal {chr(65 + i)}": name for i, name in enumerate(self.rng.sample(names, len(names)))}

    def _listing(self, labels: Dict[str, str]) -> str:
        return "\n\n".join(f"### {label}\n{self.state.proposals[name]}" for label, name in labels.items())

    def _critic_prompt(self, labels: Dict[str, str]) -> str:
        return (
            f"You are the area chair reviewing proposals for this task:\n{self.task_summary}\n\n"
            f"{self._listing(labels)}\n\n"
            "Score each proposal from 0 to 1 for scientific soundness, feasibility on one GPU, and whether its "
            'evaluation is leak-free. Return JSON: {"scores": {"<proposal label>": 0.0}, '
            '"comments": {"<proposal label>": "<the most important fix>"}}'
        )

    def _peer_prompt(self, reviewer: Expert, labels: Dict[str, str]) -> str:
        return (
            f"You are the {reviewer.name}. Review these proposals from the standpoint of {reviewer.focus}.\n\n"
            f"{self._listing(labels)}\n\nReturn JSON: "
            '{"scores": {"<proposal label>": 0.0}, "comments": {"<proposal label>": "<one concrete issue>"}}'
        )

    def run_round(self, round_index: int) -> Dict[str, Any]:
        previous_round = dict(self.state.proposals)
        for expert in self.experts:
            self.state.proposals[expert.name] = self._ask(self._proposal_prompt(expert, round_index, previous_round))

        names = list(self.state.proposals)
        critic_labels = self._labels(names)
        critic = self._ask_json(self._critic_prompt(critic_labels)) if "critic" in self.reviews else {}
        scored = critic.get("scores", {}) if isinstance(critic.get("scores"), dict) else {}
        critic_scores = {name: clamp(scored.get(label)) for label, name in critic_labels.items()}
        feedback = {critic_labels[label]: [f"Critic: {c}"] for label, c in (critic.get("comments") or {}).items()
                    if label in critic_labels}
        peer_scores: Dict[str, List[float]] = {name: [] for name in names}
        for reviewer in self.experts if "peer" in self.reviews else ():
            labels = self._labels([name for name in names if name != reviewer.name])
            review = self._ask_json(self._peer_prompt(reviewer, labels))
            for label, score in (review.get("scores") or {}).items():
                if label in labels:
                    peer_scores[labels[label]].append(clamp(score))
            for label, comment in (review.get("comments") or {}).items():
                if label in labels:
                    feedback.setdefault(labels[label], []).append(f"{reviewer.name}: {comment}")

        for name in self.state.scores:
            peers = peer_scores.get(name) or [0.5]
            # paper weights 0.4 critic and 0.3 peer; an ablated source gives its weight to the other
            parts = {"critic": (0.4, critic_scores.get(name, 0.5)), "peer": (0.3, sum(peers) / len(peers))}
            used = [parts[k] for k in ("critic", "peer") if k in self.reviews]
            review = sum(w * v for w, v in used) / sum(w for w, _ in used)
            self.state.scores[name] = 0.3 * self.state.scores[name] + 0.7 * review
        self.state.feedback = feedback
        record = {
            "round": round_index + 1,
            "proposals": dict(self.state.proposals),
            "critic": critic,
            "critic_labels": critic_labels,
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
        """The critic's editorial decision: write up the top-ranked proposal with its open reviews."""

        if not self.state.proposals:
            return None
        best = max(self.state.proposals, key=lambda n: self.state.scores.get(n, 0))
        reviews = "\n".join(f"- {line}" for line in self.state.feedback.get(best, [])) or "- none"
        prompt = (
            f"Write the research plan as the area chair. The {best}'s proposal ranked first "
            f"(coordination score {self.state.scores.get(best, 0):.2f}) and is the one selected; do not "
            f"add components from other proposals. Apply the open review points and keep its evidence IDs."
            f"\n\nTASK\n{self.task_summary}\n\nSELECTED PROPOSAL\n{self.state.proposals[best]}\n\n"
            f"OPEN REVIEWS\n{reviews}\n\nReturn only JSON with this structure:\n{json.dumps(PLAN_SCHEMA, indent=2)}"
        )
        try:
            plan = parse_json(self._ask(prompt, max_tokens=3000))
        except Exception as exc:
            print(f"Plan synthesis failed: {exc}")
            return None
        if not isinstance(plan.get("model_architecture"), dict):
            return None
        plan["selected_proposal"] = best
        return plan
