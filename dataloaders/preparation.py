"""Shared helpers for building local training index files."""

import os
from pathlib import Path

import pandas as pd

from dataloaders.config import DATA_ROOT, DATASET_CONFIG


INDEX_ROOT = Path(os.environ.get("UNID_INDEX_ROOT", str(DATA_ROOT))).expanduser()

_INDEX_SUBDIRS = {
    "hypersim": "hypersim/processed",
    "virtual_kitti_2": "virtual_kitti_2",
    "coco": "coco",
    "densepose": "coco",
    "cityscapes": "cityscapes",
    "dms": "DMS_v1",
    "point_odyssey": "point_odyssey",
    "tartanair": "tartanair",
    "spring": "spring",
    "vipseg": "VIPSeg",
}


def dataset_root(dataset, input_root=None):
    """Return an absolute source root for a preparation command."""
    if input_root is None:
        input_root = DATASET_CONFIG[dataset]["prefix"]
    return Path(input_root).expanduser().resolve()


def require_files(*paths):
    """Fail with the missing local inputs instead of relying on assertions."""
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Required dataset files are missing: " + ", ".join(missing))


def index_path(dataset, split, output_root=None):
    """Return the conventional, configurable output path for a dataset index."""
    root = Path(output_root).expanduser() if output_root is not None else INDEX_ROOT
    filename = f"{dataset}_{split}.parquet"
    return (root / _INDEX_SUBDIRS[dataset] / filename).resolve()


def write_index(rows, dataset, split, output_root=None):
    """Write a complete index atomically so interrupted runs do not corrupt it."""
    if not rows:
        raise ValueError(
            f"No rows found for {dataset} split '{split}'. Check the input root and required files."
        )
    destination = index_path(dataset, split, output_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    print(f"Wrote {len(rows)} rows to {destination}")
    return destination
