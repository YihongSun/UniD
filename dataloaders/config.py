"""Local dataset locations. Set UNID_DATA_ROOT to a directory containing datasets."""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("UNID_DATA_ROOT", "data")).expanduser()
INDEX_ROOT = Path(os.environ.get("UNID_INDEX_ROOT", str(DATA_ROOT))).expanduser()


def _dataset(name):
    return str(DATA_ROOT / name)


def _index(dataset, filename, subdir=None):
    return str(INDEX_ROOT / (subdir or dataset) / filename)


DATASET_CONFIG = {
    "hypersim": {
        "prefix": _dataset("hypersim/processed"),
        "prefix_pseudo": _dataset("hypersim"),
        "parquet_path": _index("hypersim", "hypersim_train.parquet", "hypersim/processed"),
        "is_video": False,
        "num_images": 59543,
    },
    "virtual_kitti_2": {
        "prefix": _dataset("virtual_kitti_2"),
        "prefix_pseudo": _dataset("virtual_kitti_2"),
        "parquet_path": _index("virtual_kitti_2", "virtual_kitti_2_train.parquet"),
        "is_video": False,
        "num_images": 21260,
    },
    "coco": {
        "prefix": _dataset("coco"),
        "parquet_path": _index("coco", "coco_train.parquet"),
        "is_video": False,
        "num_images": 118287,
    },
    "densepose": {
        "prefix": _dataset("coco"),
        "parquet_path": _index("densepose", "densepose_train.parquet", "coco"),
        "is_video": False,
        "num_images": 26437,
    },
    "cityscapes": {
        "prefix": _dataset("cityscapes"),
        "parquet_path": _index("cityscapes", "cityscapes_train.parquet"),
        "is_video": False,
        "num_images": 2975,
    },
    "dms": {
        "prefix": _dataset("DMS_v1"),
        "parquet_path": _index("dms", "dms_train.parquet", "DMS_v1"),
        "is_video": False,
        "num_images": 21956,
    },
    "point_odyssey": {
        "prefix": _dataset("point_odyssey"),
        "prefix_pseudo": _dataset("point_odyssey"),
        "parquet_path": _index("point_odyssey", "point_odyssey_train.parquet"),
        "is_video": True,
        "num_images": 239022,
    },
    "tartanair": {
        "prefix": _dataset("tartanair"),
        "prefix_pseudo": _dataset("tartanair"),
        "parquet_path": _index("tartanair", "tartanair_train.parquet"),
        "is_video": True,
        "num_images": 116984,
    },
    "spring": {
        "prefix": _dataset("spring"),
        "prefix_pseudo": _dataset("spring"),
        "parquet_path": _index("spring", "spring_train.parquet"),
        "is_video": True,
        "num_images": 5000,
    },
    "vipseg": {
        "prefix": _dataset("VIPSeg"),
        "parquet_path": _index("vipseg", "vipseg_train.parquet", "VIPSeg"),
        "is_video": True,
        "num_images": 66767,
    },
}
