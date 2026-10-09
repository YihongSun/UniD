from collections import defaultdict
import os, cv2, io, json, random, logging, tqdm, glob, imageio
import os.path as osp
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from panopticapi.utils import rgb2id

from dataloaders.config import DATASET_CONFIG
from dataloaders.preparation import dataset_root, require_files, write_index
from dataloaders.datasets.image_datasets.unid_img_dataset import UnidImageDataset

class ImageDMSDataset(UnidImageDataset):
    def __init__(self, resize=(480, 640), crop_scale=(0.6, 1.0), crop_ratio=(1.0,1.0), near_plane=1e-5, far_plane=80.0, full=True, fill_missing_tasks=True, load_tasks=None, **kwargs):
        super().__init__(resize=resize, crop_scale=crop_scale, crop_ratio=crop_ratio, near_plane=near_plane, far_plane=far_plane, fill_missing_tasks=fill_missing_tasks, **kwargs)
        self.config = DATASET_CONFIG["dms"]
        self.dataset_name = "dms"
        self.gt_tasks = {'rgb', 'matseg'}
        self.mat_categories, self.mat_color_code = self._get_mat_anno_info()
        self.fill_missing_tasks = fill_missing_tasks
        self.load_tasks = load_tasks if load_tasks is None else set(load_tasks)    # if None, load all available tasks

        if full:
            self.df = pd.read_parquet(self._file_io(self.config["parquet_path"]))
            logging.info(f"[DMS] found {len(self.df)} images.")
    
    def _is_gt_task(self, task_name):
        return task_name in self.gt_tasks
    
    def _is_load_task(self, task_name):
        assert self._is_gt_task(task_name), f"{task_name} not in gt tasks"
        return self.load_tasks is None or task_name in self.load_tasks

    def _get_mat_anno_info(self):
        with open(osp.join(self.config["prefix"], "taxonomy.json"), "r") as fh:
            mat_anno = json.load(fh)
        cats = {
            i: {'name': sn, 'fullname': n} for i, (n, sn) in enumerate(
                zip(mat_anno['names'], mat_anno['shortnames'])
            )}
        color_code = torch.tensor(mat_anno['colormap'])
        return cats, color_code
    
    def _load_material(self, data):
        if not self._is_load_task('matseg'):
            return None
        mat_mask_image = Image.open(self._file_io(data['mat_anno_path']))
        return mat_mask_image

def generate_parquet_file(split="train", input_root=None, output_root=None):
    DATA_DIR = str(dataset_root("dms", input_root))
    require_files(osp.join(DATA_DIR, "taxonomy.json"))

    data_list = []
    
    rgb_paths = sorted(glob.glob(osp.join(DATA_DIR, 'images', split, '*.jpg')))
    anno_paths = [x.replace('images/', 'labels/').replace('.jpg', '.png') for x in rgb_paths]
    require_files(*anno_paths)

    data_list = [{'rgb_path': x, 'mat_anno_path': y} for x, y in zip(rgb_paths, anno_paths)]
    return write_index(data_list, "dms", split, output_root)


if __name__ == "__main__":
    # generate_parquet_file()
    # sample and visualize:
    print("\nTesting image dataset...")
    image_dataset = ImageDMSDataset()
    print(f"Image dataset length: {len(image_dataset)}")

    # image_sample = image_dataset[4]

    torch.manual_seed(1)
    instances = (torch.rand(10) * len(image_dataset)).long().tolist()
    for ii in tqdm.tqdm(instances):
        x = []
        for i in range(5):
            image_sample = image_dataset[ii]
            rgb = ((image_sample['rgb'] + 1)/2).permute(1,2,0).numpy()
            material_vis = torch.zeros(rgb.shape)
            matseg = image_sample['matseg']
            material_ids = torch.unique(matseg)
            for mid in material_ids:
                material_vis[matseg==mid] = image_dataset.mat_color_code[mid.long()]
            material = material_vis.numpy()
            out_img = np.hstack((rgb, material)) * 255
            x.append(out_img)
        os.makedirs("data_vis", exist_ok=True)
        imageio.imwrite(f"data_vis/dms_sample_{ii}.png", np.vstack(x).astype(np.uint8))
