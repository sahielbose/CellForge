"""A small stand-in dataset for a smoke run of a generated script.

The verifier runs every script on it before any GPU job is launched, when
CODEGEN_SMOKE_TEST is on. It keeps the real schema (gene names, obs columns,
perturbation labels and value scale) but holds no real perturbation effect:
every cell is a resampled control cell, which every fold keeps in training, and
each stand-in perturbation group gets a synthetic shift on two genes. A script
that runs end to end on it reads the inputs the runner passes and writes
predictions in the format the scorer reads.
"""

import json
from pathlib import Path

import numpy as np

from ..dataset import is_control, perturbation_key


def build_smoke_set(dataset_file: Path, out_dir: Path, cells_per_group: int = 40, n_perturbations: int = 6,
                    seed: int = 0) -> Path:
    if (out_dir / "predict.json").exists():
        return out_dir

    import anndata as ad

    adata = ad.read_h5ad(dataset_file, backed="r")
    key = perturbation_key(adata.obs.columns)
    labels = adata.obs[key].astype(str).to_numpy()
    names = np.unique(labels)
    control_names = {name for name in names if is_control(name)}
    perturbations = sorted(set(names) - control_names)
    if not control_names or len(perturbations) < 3:
        raise ValueError("A smoke set needs control cells and at least three perturbations")

    rng = np.random.default_rng(seed)
    chosen = list(rng.choice(perturbations, size=min(n_perturbations, len(perturbations)), replace=False))
    controls = np.flatnonzero(np.isin(labels, list(control_names)))
    rows = np.sort(rng.choice(controls, size=min(len(controls), cells_per_group * (len(chosen) + 1)), replace=False))
    smoke = adata[rows].to_memory()
    adata.file.close()

    groups = np.array_split(rng.permutation(smoke.n_obs), len(chosen) + 1)
    smoke.obs[key] = smoke.obs[key].astype(str)
    X = smoke.X.tolil() if hasattr(smoke.X, "tolil") else np.array(smoke.X)
    for name, members in zip(chosen, groups[1:], strict=True):
        smoke.obs.iloc[members, smoke.obs.columns.get_loc(key)] = name
        for gene in rng.choice(smoke.n_vars, size=2, replace=False):
            X[members, gene] = X[members, gene] * 3
    smoke.X = X.tocsr() if hasattr(X, "tocsr") else X

    out_dir.mkdir(parents=True, exist_ok=True)
    held = smoke.obs[key].isin(chosen[-2:]).to_numpy()
    smoke[~held].copy().write_h5ad(out_dir / "train.h5ad")
    smoke[(smoke.obs[key] == chosen[-2]).to_numpy()].copy().write_h5ad(out_dir / "val.h5ad")
    (out_dir / "predict.json").write_text(json.dumps([chosen[-1]]), encoding="utf-8")
    (out_dir / "schema.json").write_text(json.dumps({"perturbation_key": key, "n_genes": int(smoke.n_vars)}),
                                         encoding="utf-8")
    return out_dir
