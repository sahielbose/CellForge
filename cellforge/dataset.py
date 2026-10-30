"""Read the facts the agents need from an AnnData file without loading its matrix.

Every phase used to receive the same hardcoded description (K562, CRISPRi,
scRNA-seq) whatever the dataset was. This module reads the real metadata from
the file so the analysis and the split are grounded in what is on disk.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PERTURBATION_KEYS = ("perturbation", "perturbation_id", "gene", "target_gene", "condition", "guide_identity")
CONTROL_PATTERN = ("control", "ctrl", "non-targeting", "nontargeting", "non_targeting", "unperturbed", "dmso", "vehicle")


def find_h5ad(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    files = sorted(p.glob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No .h5ad file found under {path}")
    return files[0]


def perturbation_key(columns: Iterable[str]) -> str:
    columns = set(columns)
    for key in PERTURBATION_KEYS:
        if key in columns:
            return key
    raise KeyError(f"No perturbation column in adata.obs; tried {', '.join(PERTURBATION_KEYS)}")


def is_control(label: str) -> bool:
    text = str(label).strip().lower()
    return text in {"nt", "none", "*"} or any(token in text for token in CONTROL_PATTERN)


def infer_task_type(task: str, dataset: Optional[Dict[str, Any]] = None) -> str:
    """Classify the task as genetic, chemical or cytokine perturbation.

    The specific signals are checked before the generic ones, since nearly every
    task description mentions genes.
    """

    text = f"{(dataset or {}).get('perturbation_type', '')} {task}".lower()
    if any(word in text for word in ("cytokine", "interferon", "stimulation", "ligand")):
        return "cytokine_stimulation"
    if any(word in text for word in ("drug", "compound", "chemical", "small molecule", "dose")):
        return "chemical_perturbation"
    return "genetic_perturbation"


def _most_common(obs, column: str) -> Optional[str]:
    if column not in obs.columns:
        return None
    values = obs[column].dropna().astype(str)
    return values.mode().iloc[0] if len(values) else None


def describe_dataset(path: str) -> Dict[str, Any]:
    """Summarize an .h5ad file: size, perturbation column, controls, assay hints."""

    import anndata as ad

    file = find_h5ad(path)
    adata = ad.read_h5ad(file, backed="r")
    try:
        obs = adata.obs
        info: Dict[str, Any] = {
            "dataset_name": file.stem,
            "dataset_path": str(file),
            "n_cells": int(adata.n_obs),
            "n_features": int(adata.n_vars),
            "obs_columns": list(obs.columns),
        }
        features = [str(name) for name in adata.var_names[:50]]
        peaks = sum(name.startswith("chr") and any(c in name for c in ":-_") for name in features)
        info["data_type"] = "scATAC-seq" if peaks > len(features) // 2 else "scRNA-seq"
        for column in ("perturbation_type", "cell_line", "celltype", "organism"):
            value = _most_common(obs, column)
            if value:
                info[column] = value
        try:
            key = perturbation_key(obs.columns)
        except KeyError:
            return info
        counts = Counter(obs[key].astype(str))
        controls = sorted(label for label in counts if is_control(label))
        perturbed = {label: n for label, n in counts.items() if label not in controls}
        info.update(
            perturbation_key=key,
            n_perturbations=len(perturbed),
            control_labels=controls,
            n_control_cells=sum(counts[label] for label in controls),
            median_cells_per_perturbation=int(sorted(perturbed.values())[len(perturbed) // 2]) if perturbed else 0,
            example_perturbations=sorted(perturbed, key=perturbed.get, reverse=True)[:10],
        )
        if "nperts" in obs.columns:
            info["max_targets_per_cell"] = int(obs["nperts"].max())
        return info
    finally:
        adata.file.close()
