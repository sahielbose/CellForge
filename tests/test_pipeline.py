"""Tests for the dataset reader, perturbation split, scorer, discussion and refinement."""

import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from cellforge.autorun.runner import prepare_split
from cellforge.autorun.scoring import CONTROL, group_means, n_de_genes, score_predictions
from cellforge.dataset import describe_dataset, infer_task_type, is_control
from cellforge.Method_Design import generate_research_plan
from cellforge.Method_Design.discussion import GraphDiscussion
from cellforge.Method_Design.experts import CORE, ExpertPool
from cellforge.Method_Design.main import normalize_task_analysis
from cellforge.Task_Analysis.agents import RefinementAgent
from cellforge.Task_Analysis.report import AnalysisResult

N_GENES, CELLS = 40, 30


def make_screen(path, n_perturbations=10, seed=0):
    """A small count matrix where each perturbation raises two genes of its own."""

    rng = np.random.default_rng(seed)
    labels = ["non-targeting"] * CELLS
    blocks = [rng.poisson(5, size=(CELLS, N_GENES))]
    for p in range(n_perturbations):
        rate = np.full(N_GENES, 5.0)
        rate[[2 * p % N_GENES, (2 * p + 1) % N_GENES]] = 25.0
        blocks.append(rng.poisson(rate, size=(CELLS, N_GENES)))
        labels += [f"GENE{p}"] * CELLS
    adata = ad.AnnData(np.vstack(blocks).astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(N_GENES)]
    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]
    adata.obs["perturbation"] = labels
    adata.obs["cell_line"] = "K562"
    adata.write_h5ad(path)
    return path


@pytest.fixture
def screen(tmp_path):
    return make_screen(tmp_path / "toy_screen.h5ad")


def test_describe_dataset_reads_metadata_from_file(screen):
    info = describe_dataset(str(screen))

    assert info["n_cells"] == 11 * CELLS
    assert info["n_features"] == N_GENES
    assert info["perturbation_key"] == "perturbation"
    assert info["n_perturbations"] == 10
    assert info["control_labels"] == ["non-targeting"]
    assert info["cell_line"] == "K562"


def test_split_holds_out_whole_perturbations(screen, tmp_path):
    split = prepare_split(screen, tmp_path / "split", n_folds=5, fold=0, val_ratio=0.1, seed=1)
    groups = {
        name: set(ad.read_h5ad(split[f"{name}_h5ad"]).obs["perturbation"].astype(str))
        for name in ("train", "val", "test")
    }

    assert "non-targeting" in groups["train"]
    assert not any(is_control(label) for label in groups["test"] | groups["val"])
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["val"]
    assert not groups["val"] & groups["test"]
    assert len(groups["test"]) == 2 and len(groups["val"]) == 1
    assert set(json.loads((tmp_path / "split" / "test_perturbations.json").read_text())) == groups["test"]


def test_five_folds_test_every_perturbation_once(screen, tmp_path):
    tested = []
    for fold in range(5):
        split = prepare_split(screen, tmp_path / f"fold{fold}", n_folds=5, fold=fold, seed=7)
        tested += json.loads(Path(split["test_perturbations"]).read_text())

    assert sorted(tested) == sorted(f"GENE{p}" for p in range(10))


def test_split_rejects_too_few_perturbations(tmp_path):
    with pytest.raises(ValueError, match="at least 7 perturbations"):
        prepare_split(make_screen(tmp_path / "tiny.h5ad", n_perturbations=4), tmp_path / "split")


def test_de_set_size_follows_the_paper():
    assert n_de_genes(2000) == 50
    assert n_de_genes(40) == 20
    assert n_de_genes(3) == 1


def write_predictions(path, profiles):
    labels = list(profiles)
    pred = ad.AnnData(np.vstack([profiles[label] for label in labels]).astype(np.float32))
    pred.var_names = [f"g{i}" for i in range(N_GENES)]
    pred.obs_names = labels
    pred.obs["perturbation"] = labels
    pred.write_h5ad(path)
    return str(path)


def test_perfect_predictions_beat_the_mean_shift_baseline(screen, tmp_path):
    split = prepare_split(screen, tmp_path / "split", seed=3)
    genes = [f"g{i}" for i in range(N_GENES)]
    truth = group_means(ad.read_h5ad(split["test_h5ad"]), "perturbation", genes)
    predictions = write_predictions(tmp_path / "pred.h5ad", truth)

    result = score_predictions(predictions, split["test_h5ad"], split["train_h5ad"], "perturbation")

    model = result["metrics"]["model"]
    assert result["n_scored"] == result["n_test_perturbations"] == 2
    assert model["PCC"] == pytest.approx(1.0)
    assert model["PCC_delta"] == pytest.approx(1.0)
    assert model["MSE_DE"] == pytest.approx(0.0, abs=1e-8)
    assert model["Recall"] >= 0.5
    assert model["AUROC"] > 0.9
    assert result["n_de_genes"] == 20
    assert result["metrics"]["no_change"]["PCC_delta"] == 0.0
    assert result["metrics"]["no_change"]["AUROC"] == 0.5
    assert result["metrics"]["no_change"]["Recall"] == pytest.approx(result["n_de_genes"] / N_GENES)
    assert result["beats_mean_shift_on_PCC_delta"]


def test_predicting_the_mean_shift_scores_as_the_baseline(screen, tmp_path):
    split = prepare_split(screen, tmp_path / "split", seed=3)
    genes = [f"g{i}" for i in range(N_GENES)]
    train = group_means(ad.read_h5ad(split["train_h5ad"]), "perturbation", genes)
    control = train.pop(CONTROL)
    shift = control + np.mean([m - control for m in train.values()], axis=0)
    test_labels = json.loads((tmp_path / "split" / "test_perturbations.json").read_text())
    predictions = write_predictions(tmp_path / "pred.h5ad", {label: shift for label in test_labels})

    result = score_predictions(predictions, split["test_h5ad"], split["train_h5ad"], "perturbation")

    for metric, value in result["metrics"]["mean_shift"].items():
        assert result["metrics"]["model"][metric] == pytest.approx(value, abs=1e-6)
    assert not result["beats_mean_shift_on_PCC_delta"]


def test_missing_perturbations_are_reported(screen, tmp_path):
    split = prepare_split(screen, tmp_path / "split", seed=3)
    genes = [f"g{i}" for i in range(N_GENES)]
    truth = group_means(ad.read_h5ad(split["test_h5ad"]), "perturbation", genes)
    kept = dict(list(truth.items())[:1])

    result = score_predictions(write_predictions(tmp_path / "pred.h5ad", kept), split["test_h5ad"],
                               split["train_h5ad"], "perturbation")

    assert result["n_scored"] == 1
    assert result["missing_perturbations"] == [label for label in truth if label not in kept]


class ScriptedLLM:
    """Returns a proposal for drafting prompts and fixed scores for review prompts."""

    def __init__(self, score=1.0, plan=None):
        self.score = score
        self.plan = plan
        self.calls = 0

    def chat_completion(self, messages, temperature=None, max_tokens=None, model=None):
        self.calls += 1
        prompt = messages[-1]["content"]
        names = re.findall(r"### (Proposal [A-Z])", prompt)
        if prompt.startswith("Write the research plan"):
            return json.dumps(self.plan) if self.plan else "not json"
        if "area chair" in prompt or "Review these proposals" in prompt:
            if self.score is None:
                return "no scores today"
            return json.dumps({"scores": {n: self.score for n in names}, "comments": {n: "fine" for n in names}})
        return "Use a perturbation embedding and an MLP on the change from control."


def team():
    pool = ExpertPool()
    return [pool.get(name) for name in CORE]


def test_coordination_score_follows_the_paper_update():
    discussion = GraphDiscussion(ScriptedLLM(score=1.0), team(), "task", max_rounds=1, min_rounds=1)

    discussion.run_round(0)

    # c = 0.3 * 0.5 + 0.4 * 1.0 + 0.3 * 1.0
    assert all(score == pytest.approx(0.85) for score in discussion.state.scores.values())


def test_discussion_stops_at_consensus_after_min_rounds():
    state = GraphDiscussion(ScriptedLLM(score=1.0), team(), "task", max_rounds=10, min_rounds=3).run()

    assert state.consensus_reached
    assert len(state.history) == 3


def test_discussion_stops_after_one_round_when_reviews_are_high():
    state = GraphDiscussion(ScriptedLLM(score=1.0), team(), "task", max_rounds=20).run()

    assert state.consensus_reached
    assert len(state.history) == 1


def test_first_drafts_are_independent_and_reviews_are_anonymous():
    class Recorder(ScriptedLLM):
        prompts = []

        def chat_completion(self, messages, **kwargs):
            self.prompts.append(messages[-1]["content"])
            return super().chat_completion(messages, **kwargs)

    llm = Recorder(score=0.6)
    discussion = GraphDiscussion(llm, team(), "task", max_rounds=2, min_rounds=2, seed=1)
    discussion.run_round(0)
    first = list(llm.prompts)
    discussion.run_round(1)

    drafts = [p for p in first if p.startswith("You are the") and "Round 1." in p]
    assert len(drafts) == len(CORE) and not any("OTHER PROPOSALS" in p for p in drafts)
    reviews = [p for p in llm.prompts if "area chair" in p or "Review these proposals" in p]
    assert reviews and not any(f"### {name}" in p for p in reviews for name in CORE)
    orders = [tuple(r["critic_labels"].values()) for r in discussion.state.history]
    assert orders[0] != orders[1]  # a fresh order every round


def test_scores_are_clamped_to_the_paper_range():
    discussion = GraphDiscussion(ScriptedLLM(score=0.0), team(), "task", max_rounds=1)

    discussion.run_round(0)

    # reviews of 0 count as the paper's floor of 0.1: 0.3 * 0.5 + 0.7 * 0.1
    assert all(score == pytest.approx(0.22) for score in discussion.state.scores.values())


def test_unparseable_reviews_never_reach_consensus():
    state = GraphDiscussion(ScriptedLLM(score=None), team(), "task", max_rounds=4, min_rounds=1).run()

    assert not state.consensus_reached
    assert len(state.history) == 4
    assert all(score == pytest.approx(0.5) for score in state.scores.values())


def test_failed_synthesis_falls_back_to_a_labeled_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("CELLFORGE_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("METHOD_DESIGN_MAX_ROUNDS", "3")
    analysis = {"task_description": "Predict CRISPRi responses", "dataset_info": {"dataset_name": "toy"}}

    result = generate_research_plan(analysis, output_dir=str(tmp_path / "plans"), llm=ScriptedLLM(plan=None))

    assert result["json"]["plan_source"] == "template_fallback"
    assert result["json"]["task_wise_plan"][0]["task_id"] == "T01"
    assert (tmp_path / "plans" / f"{result['generated_files']['base_filename']}.json").exists()


def test_synthesized_plan_is_marked_as_discussion_output(tmp_path, monkeypatch):
    monkeypatch.setenv("CELLFORGE_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("METHOD_DESIGN_MAX_ROUNDS", "3")
    plan = {"summary": "s", "model_architecture": {"name": "DeltaMLP", "components": []}}
    analysis = {"task_description": "Predict CRISPRi responses", "dataset_info": {"dataset_name": "toy"}}

    result = generate_research_plan(analysis, output_dir=str(tmp_path / "plans"), llm=ScriptedLLM(plan=plan))

    assert result["json"]["plan_source"] == "expert_discussion"
    assert result["json"]["model_architecture"]["name"] == "DeltaMLP"
    assert result["json"]["selected_proposal"] in CORE
    assert result["discussion_summary"]["consensus_reached"]


def test_expert_pool_normalizes_names_and_keeps_the_core_team():
    class Picker:
        def chat_completion(self, messages, **_):
            return "1. **Statistics Expert**\n- Self-critic\nnot an expert\n- Data Engineer"

    pool = ExpertPool(Picker())

    assert pool.get("  2) deep learning expert ").name == "Deep Learning Expert"
    assert pool.get("unknown person") is None
    names = [e.name for e in pool.select("genetic_perturbation", {}, size=6)]
    assert names[:4] == CORE
    assert "Self-critic" not in names
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "task, dataset, expected",
    [
        ("Predict responses to CRISPRi knockdown of genes", {}, "genetic_perturbation"),
        ("Predict gene expression after drug treatment at each dose", {}, "chemical_perturbation"),
        ("Predict gene expression after interferon stimulation", {}, "cytokine_stimulation"),
        ("Predict expression", {"perturbation_type": "small molecule"}, "chemical_perturbation"),
    ],
)
def test_infer_task_type(task, dataset, expected):
    assert infer_task_type(task, dataset) == expected


def test_normalize_task_analysis_keeps_the_full_analysis():
    report = {
        "task_description": "Report MSE_DE and PCC_DE on held-out CRISPRi perturbations",
        "dataset_info": {"dataset_name": "adamson", "n_cells": 10},
        "analysis_results": {"final_recommendations": {"model_architecture": {"family": "GNN"}}},
    }

    normalized = normalize_task_analysis(report)

    assert normalized["analysis"] == report["analysis_results"]
    assert normalized["dataset"]["name"] == "adamson"
    assert normalized["dataset"]["n_cells"] == 10
    assert normalized["evaluation_metrics"] == ["MSE_DE", "PCC_DE"]
    assert normalize_task_analysis(normalized) is normalized


def test_refinement_keeps_a_section_when_its_revision_fails():
    class Reviewer:
        def generate(self, prompt, system_prompt):
            if prompt.startswith("Review three analyses"):
                return {"content": json.dumps({"dataset": ["count the controls"], "problem": [], "baseline": []})}
            if prompt.startswith("Revise"):
                return {"content": "this is not JSON"}
            return {"content": json.dumps({"data_processing": ["log1p"]})}

    sections = [AnalysisResult({"section": name}, name) for name in ("dataset", "problem", "baseline")]

    report = RefinementAgent(Reviewer(), max_rounds=2).refine("task", {}, [], sections)

    assert report.dataset_analysis.content == {"section": "dataset"}
    assert report.final_recommendations == {"data_processing": ["log1p"]}
    assert report.refinement_comments[0] == {"dataset": ["count the controls"]}


def test_task_analysis_report_feeds_method_design(screen, tmp_path, monkeypatch):
    from cellforge.Method_Design.main import load_task_analysis
    from cellforge.Task_Analysis.main import run_task_analysis

    class AnalysisLLM:
        def generate(self, prompt, system_prompt=None, **_):
            if prompt.startswith("Review three analyses"):
                return {"content": "{}"}
            return {"content": json.dumps({"summary": "ok", "model_architecture": {"family": "delta MLP"}})}

    class Library:
        paper = {"title": "Perturb-seq", "abstract": "CRISPRi screen", "evidence_id": "cf:doi:10.1/x"}

        def search_for_task(self, task, dataset, limit=10):
            return [self.paper]

        def search_evaluation_frameworks(self, task):
            return [self.paper]

    monkeypatch.setenv("CELLFORGE_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("METHOD_DESIGN_MAX_ROUNDS", "3")
    report = run_task_analysis("Predict CRISPRi responses", describe_dataset(str(screen)), retriever=Library(),
                               llm=AnalysisLLM(), output_dir=str(tmp_path / "analyses"))
    saved = next((tmp_path / "analyses").glob("task_analysis_*.json"))

    analysis = load_task_analysis(str(saved), latest=False)
    result = generate_research_plan(analysis, output_dir=str(tmp_path / "plans"), llm=ScriptedLLM(plan=None))

    assert analysis["dataset"]["name"] == "toy_screen"
    assert analysis["dataset"]["n_perturbations"] == 10
    assert analysis["analysis"]["final_recommendations"]["model_architecture"] == {"family": "delta MLP"}
    assert result["json"]["model_architecture"] == {"family": "delta MLP"}
    assert report.dataset_analysis.metadata["evidence_ids"] == ["cf:doi:10.1/x"]


MEAN_SHIFT_SCRIPT = '''
import argparse, json, os
import anndata as ad
import numpy as np

parser = argparse.ArgumentParser()
for flag in ("--adata-path", "--val-adata-path", "--output-metrics", "--predict-perturbations", "--predictions-out"):
    parser.add_argument(flag)
args = parser.parse_args()
key = os.environ["CELLFORGE_PERTURBATION_KEY"]
train = ad.read_h5ad(args.adata_path)
x = np.log1p(train.X / train.X.sum(axis=1, keepdims=True) * 1e4)
labels = train.obs[key].astype(str).to_numpy()
control = x[labels == "non-targeting"].mean(axis=0)
shift = np.mean([x[labels == p].mean(axis=0) - control for p in set(labels) - {"non-targeting"}], axis=0)
names = json.load(open(args.predict_perturbations))
out = ad.AnnData(np.tile(control + shift, (len(names), 1)).astype(np.float32))
out.var_names = train.var_names
out.obs_names = names
out.obs[key] = names
out.write_h5ad(args.predictions_out)
json.dump({"status": "ok"}, open(args.output_metrics, "w"))
'''


def test_autorun_runs_and_scores_a_generated_script(screen, tmp_path, monkeypatch):
    import cellforge.Code_Generation as codegen
    from cellforge.autorun import run_autorun

    script = tmp_path / "result.py"
    script.write_text(MEAN_SHIFT_SCRIPT, encoding="utf-8")
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "research_plan_1.json").write_text(json.dumps({"task_wise_plan": [{"task_id": "T01"}]}))
    monkeypatch.setenv("CELLFORGE_WORKSPACE_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(codegen, "generate_code_from_plan_task",
                        lambda plan, task_id, output_dir, backend, **_: calls.append(task_id) or str(script))

    summary = run_autorun(dataset_path=str(screen), plans_dir=str(plans), fold=2)
    run_autorun(dataset_path=str(screen), plans_dir=str(plans), fold=3)

    assert calls == ["T01"]  # the second fold reuses the first fold's script

    job = summary["jobs"][0]
    assert job["status"] == "completed", job["error"]
    score = job["score"]
    assert score["n_scored"] == score["n_test_perturbations"] == 2
    # the script predicts the mean shift, so it scores exactly as the mean-shift baseline
    assert score["metrics"]["model"]["PCC_delta"] == pytest.approx(score["metrics"]["mean_shift"]["PCC_delta"], abs=1e-4)
    assert summary["split"]["fold"] == 2


def test_code_generation_phase_and_autorun_share_one_script(screen, tmp_path, monkeypatch):
    import importlib.util

    import cellforge.Code_Generation as codegen
    from cellforge.autorun import run_autorun

    spec = importlib.util.spec_from_file_location("cellforge_cli", Path(__file__).resolve().parents[1] / "main.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    script = tmp_path / "result.py"
    script.write_text(MEAN_SHIFT_SCRIPT, encoding="utf-8")
    plans = tmp_path / "data" / "plans" / "toy_screen"
    plans.mkdir(parents=True)
    (plans / "research_plan_1.json").write_text(json.dumps({"summary": "a plan with no task blocks"}))
    monkeypatch.setenv("CELLFORGE_WORKSPACE_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(codegen, "generate_code_from_plan_task",
                        lambda plan, task_id, output_dir, backend, **_: calls.append(task_id) or str(script))

    assert cli.run_code_generation({"dataset_path": str(screen)})
    summary = run_autorun(dataset_path=str(screen), fold=1)

    assert calls == [None]  # the whole plan, generated once by the phase and reused by autorun
    assert summary["jobs"][0]["status"] == "completed", summary["jobs"][0]["error"]


def test_critic_only_ablation_skips_peer_calls():
    llm = ScriptedLLM(score=1.0)
    discussion = GraphDiscussion(llm, team(), "task", max_rounds=1, reviews=("critic",))

    discussion.run_round(0)

    assert llm.calls == len(CORE) + 1  # four proposals and one critic call
    assert all(score == pytest.approx(0.85) for score in discussion.state.scores.values())
    with pytest.raises(ValueError):
        GraphDiscussion(llm, team(), "task", reviews=())


def test_baselines_can_be_scored_without_a_model(screen, tmp_path):
    split = prepare_split(screen, tmp_path / "split", seed=3)

    result = score_predictions(None, split["test_h5ad"], split["train_h5ad"], "perturbation")

    assert result["n_scored"] == 2
    assert result["metrics"]["model"] == {}
    assert result["metrics"]["mean_shift"]["PCC_delta"] > result["metrics"]["no_change"]["PCC_delta"]
    assert not result["beats_mean_shift_on_PCC_delta"]


def test_rejected_jobs_are_not_run(screen, tmp_path, monkeypatch):
    import cellforge.Code_Generation as codegen
    from cellforge.autorun import run_autorun

    script = tmp_path / "result.py"
    script.write_text("raise SystemExit('should not run')", encoding="utf-8")
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "research_plan_1.json").write_text(json.dumps({"task_wise_plan": [{"task_id": "T01"}]}))
    monkeypatch.setenv("CELLFORGE_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(codegen, "generate_code_from_plan_task", lambda plan, task_id, output_dir, backend, **_: str(script))

    summary = run_autorun(dataset_path=str(screen), plans_dir=str(plans), approve=lambda code, command: False)

    assert summary["jobs"][0]["status"] == "not_approved"
