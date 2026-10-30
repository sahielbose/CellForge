"""Tests for the smoke run, the selection split and the refinement loop."""

import json

import anndata as ad
import numpy as np
import pytest
from test_pipeline import MEAN_SHIFT_SCRIPT, make_screen

from cellforge.autorun.runner import prepare_split, smoke_set
from cellforge.autorun.smoke import build_smoke_set
from cellforge.Code_Generation.verifier import verify_generated_code

# Predicts the control profile, so its change from control is zero and PCC_delta is 0.
NO_CHANGE_SCRIPT = MEAN_SHIFT_SCRIPT.replace("control + shift", "control")
# Knows the toy rule (GENEp raises genes 2p and 2p+1), so it beats the mean shift on unseen perturbations.
ORACLE_SCRIPT = MEAN_SHIFT_SCRIPT.replace(
    "out = ad.AnnData(np.tile(control + shift, (len(names), 1)).astype(np.float32))",
    """def targets(name):
    p = int(name[4:])
    return [2 * p % x.shape[1], (2 * p + 1) % x.shape[1]]
lift = np.mean([x[labels == p].mean(axis=0)[targets(p)] - control[targets(p)] for p in set(labels) - {"non-targeting"}])
rows = []
for name in names:
    profile = control.copy()
    profile[targets(name)] += lift
    rows.append(profile)
out = ad.AnnData(np.array(rows, dtype=np.float32))""",
)


@pytest.fixture
def screen(tmp_path):
    return make_screen(tmp_path / "toy_screen.h5ad")


def test_smoke_set_keeps_the_schema_but_only_control_cells(screen, tmp_path):
    out = build_smoke_set(screen, tmp_path / "smoke", cells_per_group=5)

    full = ad.read_h5ad(screen)
    train, val = ad.read_h5ad(out / "train.h5ad"), ad.read_h5ad(out / "val.h5ad")
    predict = json.loads((out / "predict.json").read_text())
    controls = set(full.obs_names[full.obs["perturbation"] == "non-targeting"])
    assert set(train.obs_names) | set(val.obs_names) <= controls  # no real perturbed cell is used
    assert list(train.var_names) == list(full.var_names)
    assert "non-targeting" in set(train.obs["perturbation"])
    assert predict[0] not in set(train.obs["perturbation"]) | set(val.obs["perturbation"])
    assert set(train.obs["perturbation"]) - {"non-targeting"} <= set(full.obs["perturbation"])


def write_workspace(tmp_path, screen, script):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.py").write_text(script, encoding="utf-8")
    build_smoke_set(screen, workspace / "smoke", cells_per_group=5)
    return workspace


def test_smoke_run_passes_a_working_script(screen, tmp_path):
    workspace = write_workspace(tmp_path, screen, MEAN_SHIFT_SCRIPT)

    result = verify_generated_code(workspace / "result.py", workspace=workspace, smoke_dir="smoke")

    assert result.passed, [c.message for c in result.failures]
    assert {"smoke_run", "smoke_predictions"} <= {c.name for c in result.checks}


@pytest.mark.parametrize("broken, failed_check", [
    ("names = json.load(open(args.predict_perturbations))[:0]", "smoke_predictions"),
    ("names = json.load(open(args.predict_perturbations)); raise RuntimeError('shape mismatch')", "smoke_run"),
])
def test_smoke_run_catches_scripts_that_pass_help_but_fail_on_data(screen, tmp_path, broken, failed_check):
    script = MEAN_SHIFT_SCRIPT.replace("names = json.load(open(args.predict_perturbations))", broken)
    workspace = write_workspace(tmp_path, screen, script)

    result = verify_generated_code(workspace / "result.py", workspace=workspace, smoke_dir="smoke")

    assert not result.passed
    assert [c.name for c in result.failures] == [failed_check]


def test_smoke_set_is_built_only_when_enabled(screen, tmp_path, monkeypatch):
    monkeypatch.setenv("CELLFORGE_WORKSPACE_DIR", str(tmp_path))
    plan_file = tmp_path / "plans" / "toy_screen" / "research_plan_1.json"

    monkeypatch.delenv("CODEGEN_SMOKE_TEST", raising=False)
    assert smoke_set(screen, plan_file) is None
    monkeypatch.setenv("CODEGEN_SMOKE_TEST", "true")
    assert (smoke_set(screen, plan_file) / "predict.json").exists()


def test_selection_set_is_disjoint_and_mixed_into_the_prediction_list(screen, tmp_path):
    plain = prepare_split(screen, tmp_path / "plain", fold=1)
    split = prepare_split(screen, tmp_path / "select", fold=1, select_ratio=0.1)

    def labels(path):
        return set(ad.read_h5ad(path).obs["perturbation"]) - {"non-targeting"}

    select, test = labels(split["select_h5ad"]), labels(split["test_h5ad"])
    assert len(select) == 1 and not select & (labels(split["train_h5ad"]) | labels(split["val_h5ad"]) | test)
    assert json.loads(open(split["predict_perturbations"]).read()) == sorted(test | select)
    # without a selection share the split is the one autorun has always made
    assert labels(plain["val_h5ad"]) == labels(split["val_h5ad"]) and labels(plain["test_h5ad"]) == test
    assert labels(plain["train_h5ad"]) == labels(split["train_h5ad"]) | select
    assert json.loads(open(plain["predict_perturbations"]).read()) == sorted(test)


def test_refinement_keeps_only_edits_that_beat_the_best_selection_score(screen, tmp_path, monkeypatch):
    import cellforge.Code_Generation as codegen
    from cellforge.autorun import run_autorun

    scripts = {}
    for name, text in (("start", NO_CHANGE_SCRIPT), ("shift", MEAN_SHIFT_SCRIPT), ("flat", NO_CHANGE_SCRIPT),
                       ("oracle", ORACLE_SCRIPT)):
        scripts[name] = tmp_path / f"{name}.py"
        scripts[name].write_text(text, encoding="utf-8")
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "research_plan_1.json").write_text(json.dumps({"task_wise_plan": [{"task_id": "T01"}]}))
    monkeypatch.setenv("CELLFORGE_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(codegen, "generate_code_from_plan_task", lambda *a, **k: str(scripts["start"]))
    edits, reports = iter(["flat", "oracle", "shift"]), []

    def improve(plan, script, report, output_dir, **_):
        reports.append(report)
        name = next(edits)
        return str(scripts[name]), f"switched to the {name} predictor"

    monkeypatch.setattr(codegen, "improve_code", improve)

    summary = run_autorun(dataset_path=str(screen), plans_dir=str(plans), fold=1, refine_rounds=3, flush_at=[0.66])

    job = summary["jobs"][0]
    record = job["refinement"]
    assert [a["outcome"] for a in record["attempts"]] == ["discarded", "kept", "discarded"]
    assert record["final_script"] == str(scripts["oracle"]) and record["evaluations"] == 4
    assert summary["split"]["n_select_perturbations"] == 1
    assert record["selection_score"]["n_scored"] == 1
    # the journal is cleared at the flush point (round 2) and grows again afterwards
    assert "round 1" not in reports[1] and "round 2" in reports[2]
    # the test fold is scored once, for the final script, and it beats the mean shift
    assert job["score"]["n_scored"] == 2 and job["score"]["beats_mean_shift_on_PCC_delta"]


def test_refinement_needs_a_selection_set(screen, tmp_path):
    from cellforge.autorun import run_autorun

    with pytest.raises(ValueError, match="selection"):
        run_autorun(dataset_path=str(screen), plans_dir=str(tmp_path), refine_rounds=1, select_ratio=0)


def test_oracle_script_is_better_than_the_mean_shift(screen, tmp_path):
    """Guards the refinement test: its 'better' edit must really be better on unseen perturbations."""

    import subprocess
    import sys

    from cellforge.autorun.runner import build_command
    from cellforge.autorun.scoring import score_predictions

    split = prepare_split(screen, tmp_path / "split", fold=1)
    script = tmp_path / "oracle.py"
    script.write_text(ORACLE_SCRIPT, encoding="utf-8")
    command = build_command(script, split, tmp_path, sys.executable)
    subprocess.run(command, shell=True, check=True, env={"CELLFORGE_PERTURBATION_KEY": "perturbation",
                                                         "PATH": "/usr/bin:/bin"})
    score = score_predictions(str(tmp_path / "predictions.h5ad"), split["test_h5ad"], split["train_h5ad"],
                              "perturbation")
    assert score["metrics"]["model"]["PCC_delta"] > score["metrics"]["mean_shift"]["PCC_delta"] + 0.1
    assert np.isfinite(score["metrics"]["model"]["MSE_DE"])
