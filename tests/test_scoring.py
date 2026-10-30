"""Tests for the linear baseline, the perturbation-specific metrics and the Wilcoxon DE ranking."""

import anndata as ad
import numpy as np
import pytest
from scipy.stats import ranksums
from test_pipeline import N_GENES, make_screen, write_predictions

from cellforge.autorun.runner import prepare_split
from cellforge.autorun.scoring import (
    CONTROL,
    _normalized,
    discrimination,
    gene_targets,
    group_means,
    linear_baseline,
    score_predictions,
    wilcoxon_z,
)

MODULES, PER_MODULE = 4, 10


def make_module_screen(path, seed=0):
    """Forty genes in four co-regulated modules; activating any gene raises its whole module."""

    rng = np.random.default_rng(seed)
    targets = [m * PER_MODULE + j for m in range(MODULES) for j in range(4)]
    labels, blocks = ["non-targeting"] * 40, [rng.poisson(5, size=(40, N_GENES))]
    for t in targets:
        rate = np.full(N_GENES, 5.0)
        rate[t // PER_MODULE * PER_MODULE:(t // PER_MODULE + 1) * PER_MODULE] = 15.0
        blocks.append(rng.poisson(rate, size=(40, N_GENES)))
        labels += [f"g{t}"] * 40
    adata = ad.AnnData(np.vstack(blocks).astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(N_GENES)]
    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]
    adata.obs["perturbation"] = labels
    adata.write_h5ad(path)
    return path


def test_linear_model_predicts_the_module_of_an_unseen_gene(tmp_path):
    screen = make_module_screen(tmp_path / "modules.h5ad")
    split = prepare_split(screen, tmp_path / "split", fold=0)

    result = score_predictions(None, split["test_h5ad"], split["train_h5ad"], "perturbation")

    linear, shift = result["metrics"]["linear"], result["metrics"]["mean_shift"]
    assert result["linear_coverage"] == 1.0
    assert linear["PCC_delta"] > shift["PCC_delta"] + 0.3
    assert linear["PCC_delta_pert"] > 0.5 and shift["PCC_delta_pert"] == 0.0
    assert linear["Discrimination"] > shift["Discrimination"]


def test_linear_model_adds_combinations_and_skips_unmeasured_targets():
    genes = [f"g{i}" for i in range(6)]
    control = np.ones(6)
    effects = {f"g{i}": np.eye(6)[i] * (i + 1) for i in range(4)}
    stats = {CONTROL: (control, None, 10)} | {label: (control + e, None, 10) for label, e in effects.items()}

    predict = linear_baseline(stats, control, genes, k=3)

    assert np.allclose(predict("g0+g2"), control + effects["g0"] + effects["g2"])
    assert predict("g5") is not None and predict("DMSO_drug") is None and predict("g0+drug") is None
    assert gene_targets("g1_g3", {g: i for i, g in enumerate(genes)}) == [1, 3]


def test_discrimination_is_one_when_perfect_and_near_half_when_constant():
    rng = np.random.default_rng(0)
    observed = rng.normal(size=(10, 30))
    masks = [np.ones(30, dtype=bool)] * 10

    assert np.all(discrimination(observed, observed, masks) == 1)
    constant = discrimination(np.zeros_like(observed), observed, masks)
    assert constant.mean() == pytest.approx(11 / 20)  # (N + 1) / 2N for N = 10


def test_perfect_predictions_are_perturbation_specific(tmp_path):
    screen = make_screen(tmp_path / "toy_screen.h5ad")
    split = prepare_split(screen, tmp_path / "split", seed=3)
    truth = group_means(ad.read_h5ad(split["test_h5ad"]), "perturbation", [f"g{i}" for i in range(N_GENES)])

    result = score_predictions(write_predictions(tmp_path / "pred.h5ad", truth), split["test_h5ad"],
                               split["train_h5ad"], "perturbation")

    assert result["metrics"]["model"]["PCC_delta_pert"] == pytest.approx(1.0)
    assert result["metrics"]["model"]["Discrimination"] == 1.0
    assert result["metrics"]["mean_shift"]["PCC_delta_pert"] == 0.0
    assert result["linear_coverage"] == 0.0  # GENEp names no measured gene, so the linear row is the mean shift
    assert result["metrics"]["linear"] == result["metrics"]["mean_shift"]


def test_wilcoxon_z_matches_the_rank_sum_statistic():
    rng = np.random.default_rng(1)
    perturbed, control = rng.normal(0.5, 1, size=(25, 8)), rng.normal(0, 1, size=(60, 8))

    expected = [ranksums(perturbed[:, g], control[:, g]).statistic for g in range(8)]

    assert np.allclose(wilcoxon_z(perturbed, control), expected)


def test_wilcoxon_de_ranking_finds_the_raised_genes(tmp_path):
    screen = make_screen(tmp_path / "toy_screen.h5ad")
    split = prepare_split(screen, tmp_path / "split", seed=3)
    truth = group_means(ad.read_h5ad(split["test_h5ad"]), "perturbation", [f"g{i}" for i in range(N_GENES)])
    predictions = write_predictions(tmp_path / "pred.h5ad", truth)

    result = score_predictions(predictions, split["test_h5ad"], split["train_h5ad"], "perturbation",
                               de_test="wilcoxon")

    assert result["de_test"] == "wilcoxon"
    assert result["metrics"]["model"]["MSE_DE"] == pytest.approx(0.0, abs=1e-8)
    test, train = ad.read_h5ad(split["test_h5ad"]), ad.read_h5ad(split["train_h5ad"])
    genes = [f"g{i}" for i in range(N_GENES)]
    for label in set(test.obs["perturbation"]):
        z = wilcoxon_z(_normalized(test, "perturbation", genes, label), _normalized(train, "perturbation", genes, CONTROL))
        p = int(label[4:])
        assert set(np.argsort(-np.abs(z))[:2]) == {2 * p % N_GENES, (2 * p + 1) % N_GENES}
    with pytest.raises(ValueError, match="de_test"):
        score_predictions(predictions, split["test_h5ad"], split["train_h5ad"], "perturbation", de_test="anova")


def test_report_collects_folds_into_one_results_page(tmp_path, monkeypatch):
    import json

    from test_pipeline import MEAN_SHIFT_SCRIPT

    import cellforge.Code_Generation as codegen
    from cellforge.autorun import run_autorun
    from cellforge.autorun.report import collect, render

    screen = make_screen(tmp_path / "toy_screen.h5ad")
    script = tmp_path / "result.py"
    script.write_text(MEAN_SHIFT_SCRIPT, encoding="utf-8")
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "research_plan_1.json").write_text(json.dumps({"task_wise_plan": [{"task_id": "T01"}]}))
    monkeypatch.setenv("CELLFORGE_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(codegen, "generate_code_from_plan_task", lambda *a, **k: str(script))
    for fold in (0, 1):
        run_autorun(dataset_path=str(screen), plans_dir=str(plans), fold=fold)

    rows = collect([str(tmp_path)])
    page = render(rows)

    assert [j["fold"] for j in rows["toy_screen"]] == [0, 1]
    assert "## toy_screen" in page and "on folds 0, 1" in page
    assert "beat the mean shift on PCC_delta in 0 of 2" in page  # it predicts the mean shift itself
    assert "| PCC on change from control | " in page and "| Discrimination | " in page
