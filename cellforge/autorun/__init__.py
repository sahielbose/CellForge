"""Autorun: perturbation-level splits, execution (local or Slurm), and held-out scoring."""

from .runner import prepare_split, run_autorun
from .scoring import score_predictions, score_run

__all__ = ["prepare_split", "run_autorun", "score_predictions", "score_run"]
