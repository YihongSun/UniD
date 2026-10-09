"""Build a UniD training index from an already prepared local dataset tree."""

import argparse
import importlib
from pathlib import Path


BUILDERS = {
    "coco": "dataloaders.datasets.image_datasets.coco",
    "cityscapes": "dataloaders.datasets.image_datasets.cityscapes",
    "densepose": "dataloaders.datasets.image_datasets.densepose",
    "dms": "dataloaders.datasets.image_datasets.dms",
    "hypersim": "dataloaders.datasets.image_datasets.hypersim",
    "virtual_kitti_2": "dataloaders.datasets.image_datasets.virtual_kitti_2",
    "point_odyssey": "dataloaders.datasets.video_datasets.point_odyssey",
    "tartanair": "dataloaders.datasets.video_datasets.tartanair",
    "spring": "dataloaders.datasets.video_datasets.spring",
    "vipseg": "dataloaders.datasets.video_datasets.vipseg",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(BUILDERS))
    parser.add_argument("--split", default="train", help="Dataset split to index (default: train)")
    parser.add_argument("--input-root", required=True, type=Path, help="Root of the extracted, prepared dataset")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Root for generated indexes (default: UNID_INDEX_ROOT, otherwise UNID_DATA_ROOT, default data/)",
    )
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    if not input_root.is_dir():
        parser.error(f"input root does not exist or is not a directory: {input_root}")

    module = importlib.import_module(BUILDERS[args.dataset])
    try:
        module.generate_parquet_file(
            split=args.split,
            input_root=input_root,
            output_root=args.output_root.expanduser().resolve() if args.output_root else None,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
