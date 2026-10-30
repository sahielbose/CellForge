# GPU benchmark plan

This plan measures the rewritten pipeline against the numbers in the paper, and then tests changes that could improve it, each at the same compute budget on held-out perturbations. Every choice between candidates (a design, a prompt, a branch) is made on the validation perturbations, and the test fold is scored once per condition at the end. Every condition also gets the same number of training evaluations and the same GPU hours on a given dataset, since a method that wins only because it ran longer has not been compared fairly.

## Setup

The paper trained each model on one GPU per Slurm job with a one-day wall-time limit, and its designed models have 10 to 30 million parameters, so any 40 to 80 GB data-center GPU (A100, H100, L40S) is enough. Srivatsan is a 2.4 GB file and its jobs need more host memory than the others.

```bash
pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cu124   # generated scripts use PyTorch; match your CUDA
npm install -g @openai/codex && codex login
python scripts/download_datasets.py adamson norman
cellforge --init && cellforge --doctor
```

Fix `MODEL_NAME` and `CODEX_MODEL` in `.env` before the first run and keep them for every condition, and write the model versions, the commit of this repository and the GPU type into the results table. For a faithful reproduction, set `METHOD_DESIGN_MAX_ROUNDS=20`, the paper's cap, since the default here is six to keep exploratory runs cheap. Task Analysis and Method Design save their token use under `llm_usage` in the report and the plan, which is what the equal-budget comparisons count.

Start with Adamson (`AdamsonWeissman2016_GSM2406681_10X010.h5ad`) and Norman, the two datasets with headline numbers in the paper, and extend to all six once the pipeline is stable.

## Stage 0: reference numbers, no LLM

The harness scores the no-change and mean-shift predictors on any split, which gives the floor every designed model has to clear and takes a few minutes on a CPU.

```python
from pathlib import Path
from cellforge.autorun import prepare_split, score_predictions

data = Path("data/datasets/AdamsonWeissman2016_GSM2406681_10X010.h5ad")
for k in range(5):
    split = prepare_split(data, data.parent / "splits" / "reference" / f"fold{k}", fold=k)
    result = score_predictions(None, split["test_h5ad"], split["train_h5ad"], split["perturbation_key"])
    print(k, result["metrics"]["mean_shift"])
```

The fold assignment depends only on the seed, so these folds are the same ones autorun builds. `python -m cellforge.autorun.scoring --baselines <split folder>` does the same for a single existing split.

## Stage 1: reproduce the paper (C0)

Run the full pipeline three times per dataset, as the paper did, and train each resulting design on all five folds, which makes 15 training jobs per dataset.

```bash
DATA=data/datasets/AdamsonWeissman2016_GSM2406681_10X010.h5ad
for run in 1 2 3; do
  cellforge --dataset-path $DATA
  for k in 0 1 2 3 4; do
    cellforge --phase autorun --dataset-path $DATA --fold $k --executor slurm \
              --partition gpu --gres gpu:1 --mem 64G --slurm-time 24:00:00
  done
done
python -m cellforge.autorun.scoring data/runs/<dataset>/<run folder>   # after each set of jobs
```

Report the fold-averaged metrics as a mean and standard deviation over the three runs, next to the two reference predictors and the paper's figures (PCC 0.9883 on Adamson, MSE on DE genes 0.1736 on Norman). Absolute MSE depends on normalization, so compare MSE within a dataset and treat PCC on the change from control as the main cross-check, because the mean-shift baseline already scores well on plain PCC. The first number to send back from this stage is the measured time per training job, since it sets the budget for everything after it.

## Stage 2: what the discussion contributes

The paper says it has not measured whether the critic-mediated discussion changes the designs, and three runs per dataset cannot show it. These conditions isolate each part of the ideation stage with switches that already exist. Each one is three full runs on five folds, the same as C0.

| Condition | Setting | Question |
|---|---|---|
| A1 | `METHOD_DESIGN_REVIEWS=critic` | does peer review add anything beyond the critic? |
| A2 | `METHOD_DESIGN_REVIEWS=peer` | does the central critic add anything beyond peers? |
| A3 | `METHOD_DESIGN_MAX_ROUNDS=1` | is one round of proposals as good as a discussion? |
| A4 | `METHOD_DESIGN_MIN_ROUNDS=3` | does forcing revision help when the reviews start high? |
| A5 | `CELLFORGE_ONLINE_RETRIEVAL=false` and an empty `CELLFORGE_LITERATURE_DIR` | how much do the retrieved papers matter? |
| A6 | `METHOD_DESIGN_MAX_EXPERTS=2` and `=6` | does team size matter? |
| A7 | `TASK_ANALYSIS_MAX_REFINEMENT_ROUNDS=0` | does refining the task analysis matter? |

Besides the test metrics, record the tokens used and the architecture each run chose (`model_architecture` in the plan JSON). If the designs barely differ across conditions, the discussion is not what drives the results, and that would be worth reporting on its own.

## Stage 3: search with real scores

These conditions follow the lab's action plan. They replace "discuss, then build once" with a search that uses validation scores, and they need code that does not exist yet. The comparison runs on Adamson and Norman with at least three repeats each, and every condition gets the same number of evaluations.

| Condition | What runs | Question |
|---|---|---|
| C1 | one coding agent in a closed loop: train, read the validation score and diagnostics, edit, and keep the edit only if it beats the best score by more than run-to-run noise | does feedback alone help? |
| C2 | four independent C1 runs, best kept | is any gain from C1 just more samples? |
| C3 | Fork-and-Flush on C1: four starting agents, the best becomes the leader, and at 33% and 66% of the budget the leader is forked into four branches that keep the files but start with an empty conversation, each with about 6% of the budget | does escaping a fixed line of ideas help? |
| C4 | a team of four agents that share candidates as git commits with their scores, with three forgets per run that clear every session and restart all agents from the best candidate | do shared scores and forgetting help? |
| C5 | the better of C3 and C4, with prompts tuned by GEPA's `optimize_anything`, the validation score as the evaluator, and each readout as its own subset of the Pareto frontier, tuned on some datasets and tested on others | do tuned prompts add anything? |

The settings for C3 and C4 come from the Fork-and-Flush paper (arXiv 2610.07447) and the forgetting paper (arXiv 2610.20563). As a sizing example from the action plan, C1 to C5 on two datasets with three repeats and 100 evaluations each is 3,000 evaluations, about 250 GPU hours at five minutes per evaluation and 750 at fifteen.

## Changes still to build

Ordered by how much each would change what the benchmark can say.

| Change | Why | Size |
|---|---|---|
| linear and random-forest predictors owned by the harness, including the linear model on gene embeddings from Ahlmann-Eltze, Huber and Anders (*Nature Methods* 2025) | they are part of the paper's reference set, and a designed model that does not beat a linear model has not justified its size | small; needs a check of how each dataset names its targets |
| a verifier run on a small subsample before any GPU job is launched | today the verifier only compiles the script and reads its `--help`, while the paper's review attributes 64% of failures to execution errors and invalid types | small |
| the closed loop (C1) inside autorun | every later condition builds on it | medium |
| Fork-and-Flush and the forgetting team (C3, C4), sharing one store of scored commits | the main search methods to compare | medium |
| GEPA prompt tuning (C5) | the third search method | medium |
| the discussion's plan as one starting candidate among the published models and the baselines, with a fixed budget in place of the consensus rule | lets measured scores, not the critic, pick the design | medium |
| Wilcoxon ranking of DE genes for the multimodal readouts | matches the paper's protocol exactly for Papalexi and Liscovitch-Brauer | small |
| a map of design diversity across runs, built from the plans and scripts already generated | answers the paper's open question about whether the discussion diversifies designs; needs only LLM calls | small |
| one dataset published after the language model's training cutoff | separates design ability from memory of published models | depends on the dataset |
