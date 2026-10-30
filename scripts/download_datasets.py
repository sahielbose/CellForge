#!/usr/bin/env python3
"""Download the six benchmark datasets as scPerturb AnnData files from Zenodo.

RNA and protein files come from record 13350497 and ATAC files from 7058382.
Files already present with the published MD5 are skipped.

    python scripts/download_datasets.py --list
    python scripts/download_datasets.py adamson norman --out data/datasets
    python scripts/download_datasets.py --all --dry-run
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

RNA, ATAC = "13350497", "7058382"

# key: (description, GEO accession, [(file, record, bytes, md5)])
DATASETS = {
    "adamson": ("CRISPRi, scRNA-seq", "GSE90546", [
        ("AdamsonWeissman2016_GSM2406675_10X001.h5ad", RNA, 34557246, "232f7e3756d41602bbe434b50662a76f"),
        ("AdamsonWeissman2016_GSM2406677_10X005.h5ad", RNA, 139059637, "8657391920e7f8b3e6fd52745777002a"),
        ("AdamsonWeissman2016_GSM2406681_10X010.h5ad", RNA, 471286951, "2fa44ea61a8dd35742af618638ec65fc"),
    ]),
    "norman": ("CRISPRa single and paired, scRNA-seq", "GSE133344", [
        ("NormanWeissman2019_filtered.h5ad", RNA, 698680199, "c870e6967d91c017d9da827bab183cd6"),
    ]),
    "srivatsan": ("small molecules at several doses (sci-Plex 3), scRNA-seq", "GSE139944", [
        ("SrivatsanTrapnell2020_sciplex3.h5ad", RNA, 2526631614, "c9e70629505d98c7ca1a837f62b14e89"),
    ]),
    "schiebinger": ("time course of mouse fibroblast reprogramming, scRNA-seq", "GSE106340", [
        ("SchiebingerLander2019_GSE106340.h5ad", RNA, 378858726, "8d988c345930195ea6e64a05c762abb3"),
    ]),
    "papalexi": ("CRISPR, ECCITE-seq RNA and surface protein", "GSE153056", [
        ("PapalexiSatija2021_eccite_RNA.h5ad", RNA, 147215278, "d9e8bfe20b0e7b0919be042e8fcf6d03"),
        ("PapalexiSatija2021_eccite_protein.h5ad", RNA, 1191551, "07290242b0e835c9474bb816de9cda45"),
    ]),
    "liscovitch": ("CRISPR, scATAC-seq (zipped)", "GSE168851", [
        ("Liscovitch-BrauerSanjana2021_K562_1.zip", ATAC, 202476643, "43c1ac2e1cfc0294a79ecc8c452726b5"),
        ("Liscovitch-BrauerSanjana2021_K562_2.zip", ATAC, 347307705, "4cadbeec133fa071f3aeffcabdf71867"),
    ]),
}


def size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - Zenodo publishes MD5
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, record: str, expected: str, dest: Path) -> bool:
    if dest.exists() and md5(dest) == expected:
        print(f"  {name}: present, checksum ok")
        return True
    url = f"https://zenodo.org/records/{record}/files/{name}?download=1"
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  {name}: downloading")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "cellforge"})
        with urllib.request.urlopen(request) as response, part.open("wb") as handle:  # noqa: S310
            shutil.copyfileobj(response, handle, 1 << 20)
    except OSError as exc:
        print(f"  {name}: failed ({exc})", file=sys.stderr)
        part.unlink(missing_ok=True)
        return False
    if md5(part) != expected:
        print(f"  {name}: checksum mismatch, removed", file=sys.stderr)
        part.unlink()
        return False
    part.replace(dest)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", metavar="dataset", help=", ".join(DATASETS))
    parser.add_argument("--all", action="store_true", help="download all six")
    parser.add_argument("--list", action="store_true", help="list the datasets and exit")
    parser.add_argument("--out", default="data/datasets", help="destination folder")
    parser.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    args = parser.parse_args()
    unknown = sorted(set(args.datasets) - set(DATASETS))
    if unknown:
        parser.error(f"unknown dataset {', '.join(unknown)}; choose from {', '.join(DATASETS)}")

    if args.list or not (args.datasets or args.all):
        for key, (description, accession, files) in DATASETS.items():
            print(f"{key:<12} {description}; {accession}; {size(sum(f[2] for f in files))}")
        return 0

    keys = list(DATASETS) if args.all else list(dict.fromkeys(args.datasets))
    files = [f for key in keys for f in DATASETS[key][2]]
    out = Path(args.out).expanduser().resolve()
    needed = sum(f[2] for f in files)
    print(f"{len(files)} file(s), {size(needed)}, into {out}")
    if args.dry_run:
        for name, record, n, _ in files:
            print(f"  {name}  {size(n)}  https://zenodo.org/records/{record}/files/{name}")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < needed * 1.1:
        print("Not enough free disk space", file=sys.stderr)
        return 1
    failed = [name for name, record, _, digest in files if not fetch(name, record, digest, out / name)]
    if failed:
        print(f"{len(failed)} file(s) failed; re-run to retry: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
