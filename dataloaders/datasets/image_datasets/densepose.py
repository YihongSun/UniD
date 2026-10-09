from collections import defaultdict
import os, cv2, io, json, random, logging, tqdm, glob, imageio
import os.path as osp
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pycocotools.mask as MaskUtils

from dataloaders.config import DATASET_CONFIG
from dataloaders.preparation import dataset_root, require_files, write_index
from dataloaders.datasets.utils import generate_gaussian_heatmaps
from dataloaders.transforms.spatial_transform import SpatialTransform
from dataloaders.datasets.image_datasets.unid_img_dataset import UnidImageDataset


class ImageDensePoseDataset(UnidImageDataset):
    def __init__(self, resize=(480, 640), crop_scale=(0.6, 1.0), crop_ratio=(1.0,1.0), near_plane=1e-5, far_plane=80.0, full=True, fill_missing_tasks=True, load_tasks=None, **kwargs):
        super().__init__(resize=resize, crop_scale=crop_scale, crop_ratio=crop_ratio, near_plane=near_plane, far_plane=far_plane, fill_missing_tasks=fill_missing_tasks, **kwargs)
        self.config = DATASET_CONFIG["densepose"]
        self.dataset_name = "densepose"
        self.gt_tasks = {'rgb', 'partseg-coco'}
        self.load_tasks = load_tasks if load_tasks is None else set(load_tasks)    # if None, load all available tasks

        if full:
            self.df = pd.read_parquet(self._file_io(self.config["parquet_path"]))
            self.imgid2anno = self._load_coco_annotations()
            logging.info(f"[ImageDensePoseDataset] found {len(self.df)} images.")
    
    def _is_gt_task(self, task_name):
        return task_name in self.gt_tasks
    
    def _is_load_task(self, task_name):
        assert self._is_gt_task(task_name), f"{task_name} not in gt tasks"
        return self.load_tasks is None or task_name in self.load_tasks
    
    def _load_coco_annotations(self):
        densepose_anno_path = osp.join(
            self.config["prefix"], 'annotations', 'densepose_coco_2014_train.json'
        )
        densepose_anno = json.load(self._file_io(densepose_anno_path))

        imgid2anno = defaultdict(list)
        for anno in densepose_anno['annotations']:
            imgid2anno[anno['image_id']].append(anno)

        return dict(imgid2anno)
    
    def _load_part_coco(self, data):
        if not self._is_load_task('partseg-coco'):
            return None
        image_id = data['image_id']
        annos = self.imgid2anno.get(image_id, [])
        height, width = data['height'], data['width']

        partseg = np.zeros((height, width), dtype=np.uint8)
        annos.sort(key=lambda x: int('dp_masks' in x), reverse=True)    # first load annotations w/o densepose masks
        for ann in annos:
            if ann['category_id'] != 1:   # only person class
                continue
            Polys = ann.get('dp_masks', None)
            if Polys is None:
                ignore_mask = MaskUtils.decode(MaskUtils.frPyObjects(ann['segmentation'], height, width))[...,0]
                partseg[ignore_mask>0] = 15 # mark ignored regions
                continue

            local_partseg = np.zeros([256,256])
            for i in range(1,15):
                if(Polys[i-1]):
                    current_mask = MaskUtils.decode(Polys[i-1])
                    local_partseg[current_mask>0] = i
            
            bbr =  np.array(ann['bbox']).astype(int)
            x1,y1,x2,y2 = bbr[0],bbr[1],bbr[0]+bbr[2],bbr[1]+bbr[3]
            x2 = min( [ x2,partseg.shape[1] ] );  y2 = min( [ y2,partseg.shape[0] ] )

            ann_partseg = cv2.resize(local_partseg, (int(x2-x1),int(y2-y1)) ,interpolation=cv2.INTER_NEAREST)
            partseg[y1:y2, x1:x2][ann_partseg>0] = ann_partseg[ann_partseg>0]

        return Image.fromarray(partseg)

def generate_parquet_file(split="train", input_root=None, output_root=None):
    DATA_DIR = str(dataset_root("densepose", input_root))

    img_dir = osp.join(DATA_DIR, f'{split}2014')
    anno_path = osp.join(DATA_DIR, 'annotations', f'densepose_coco_2014_{split}.json')
    require_files(anno_path)
    with open(anno_path, 'r') as f:
        anno = json.load(f)
        images = sorted(anno['images'], key=lambda image: image['id'])
    
    data_list = []
    for image_anno in tqdm.tqdm(images, desc="Processing annotated images"):
        image_id = image_anno['id']
        image_path = osp.join(img_dir, image_anno['file_name'])
        if not osp.isfile(image_path):
            raise FileNotFoundError(f"Required image is missing: {image_path}")

        data_list.append({
            'image_id': image_id,
            'rgb_path': image_path,
            'height': image_anno['height'],
            'width': image_anno['width'],
        })

    return write_index(data_list, "densepose", split, output_root)


if __name__ == "__main__":
    # generate_parquet_file()

    # sample and visualize:
    print("\nTesting image dataset...")
    image_dataset = ImageDensePoseDataset()
    print(f"Image dataset length: {len(image_dataset)}")

    # image_sample = image_dataset[4]

    torch.manual_seed(1); rand_proj = torch.rand(16,3); rand_proj[0,:] = 0.0; rand_proj[-1,:] = 1.0
    instances = (torch.rand(10) * len(image_dataset)).long().tolist()
    for ii in tqdm.tqdm(instances):
        x = []
        for i in range(5):
            image_sample = image_dataset[ii]
            rgb = ((image_sample['rgb'] + 1)/2).permute(1,2,0).numpy()
            partseg = image_sample['partseg-coco']
            partseg_vis = partseg[:,:,np.newaxis].repeat((1,1,3)) * 10 / 255
            # partseg_vis = rand_proj[partseg].numpy()
            out_img = np.hstack((rgb, partseg_vis)) * 255
            x.append(out_img)
        os.makedirs("data_vis", exist_ok=True)
        imageio.imwrite(f"data_vis/densepose_sample_{ii}.png", np.vstack(x).astype(np.uint8))
