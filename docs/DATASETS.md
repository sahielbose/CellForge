# Datasets

The paper benchmarks CellForge on six perturbation datasets, all distributed as harmonized AnnData files by [scPerturb](https://projects.sanderlab.org/scperturb/) (Peidli et al., *Nature Methods* 2024). The RNA and protein files are in Zenodo record [13350497](https://doi.org/10.5281/zenodo.13350497) and the ATAC files in record [7058382](https://doi.org/10.5281/zenodo.7058382). `scripts/download_datasets.py` fetches them and checks every file against the MD5 that Zenodo publishes.

| Key | Study | Perturbation | Readout | GEO | Download |
|---|---|---|---|---|---:|
| `adamson` | Adamson et al. 2016 | CRISPRi, single genes | scRNA-seq | GSE90546 | 615 MB |
| `norman` | Norman et al. 2019 | CRISPRa, single genes and pairs | scRNA-seq | GSE133344 | 666 MB |
| `srivatsan` | Srivatsan et al. 2020 (sci-Plex 3) | small molecules at several doses | scRNA-seq | GSE139944 | 2.4 GB |
| `schiebinger` | Schiebinger et al. 2019 | time course of mouse fibroblast reprogramming, which the paper treats as cytokine-driven differentiation | scRNA-seq | GSE106340 | 361 MB |
| `papalexi` | Papalexi et al. 2021 (ECCITE-seq) | CRISPR | RNA and surface protein, scored separately | GSE153056 | 142 MB |
| `liscovitch` | Liscovitch-Brauer et al. 2021 | CRISPR | scATAC-seq | GSE168851 | 524 MB |

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py adamson norman      # into data/datasets
python scripts/download_datasets.py --all --dry-run     # file names, sizes and URLs only
```

Adamson comes as three files, one per GEO sample (GSM2406675, GSM2406677 and GSM2406681), and the paper does not say which it used, so the quick start points at the largest. The Liscovitch-Brauer data arrives as two zip archives that have to be unpacked before use.

## What CellForge expects in a file

Any `.h5ad` works if it has these two things.

The perturbation label is read from the first of these `obs` columns that exists: `perturbation`, `perturbation_id`, `gene`, `target_gene`, `condition`, `guide_identity`, and the scPerturb files use `perturbation`.

Control cells are recognized by their label. A label counts as a control if it contains `control`, `ctrl`, `non-targeting`, `nontargeting`, `non_targeting`, `unperturbed`, `dmso` or `vehicle`, or is exactly `nt`, `none` or `*`, ignoring case. All control labels are pooled into one reference group, and a dataset with no control cells cannot be split, because every prediction is made relative to them.

`X` can hold raw counts or log-normalized values. When it holds integers, the scorer normalizes each cell to 10,000 counts over all genes and takes log1p before averaging, and when it does not, the values are used as they are. Generated models are asked to write predictions on the log1p(CP10k) scale either way. The feature space is called scATAC-seq when most variable names look like genomic coordinates (`chr...`) and scRNA-seq otherwise.

Task Analysis records the facts it reads in its report: cell and feature counts, the perturbation column, the number of perturbations, the control labels and the number of control cells, the median number of cells per perturbation, examples of perturbation names, the largest number of targets in one cell when there is an `nperts` column, and the `cell_line`, `celltype`, `organism` and `perturbation_type` annotations when they exist.

## Splits

`cellforge --phase autorun --fold k` writes the split for fold `k` under `data/datasets/splits/<dataset>/`, and a later run with the same fold, fold count, validation share and seed reuses it. The perturbation identities that are not controls are shuffled with the seed (42 by default) and dealt into `--folds` groups (five by default). Fold `k` is the test set. Validation takes 10% of all perturbations (`--split-val-ratio`) from those left after the test fold, and everything else, plus every control cell, is training. With `--select-ratio` (on by default at 10% when `--refine-rounds` is set) a further share becomes a selection set that the refinement loop scores but the script never receives, and its folder name gains a `_sel` suffix so it never overwrites a plain split. A dataset needs at least seven non-control perturbations for five folds. Each split folder holds `train.h5ad`, `val.h5ad`, `test.h5ad`, `select.h5ad` when there is a selection set, the lists of held-out names, the list of names the script must predict, and a `split_manifest.json` with the counts and settings.
