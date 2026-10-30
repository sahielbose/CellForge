"""Autorun: perturbation-level splits, execution (local or Slurm), and held-out scoring."""

__all__ = ["prepare_split", "run_autorun", "score_predictions", "score_run"]


def __getattr__(name):
    # imported lazily so `python -m cellforge.autorun.scoring` does not load the module twice
    if name in ("prepare_split", "run_autorun"):
        from . import runner
        return getattr(runner, name)
    if name in ("score_predictions", "score_run"):
        from . import scoring
        return getattr(scoring, name)
    raise AttributeError(name)
