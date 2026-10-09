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
from dataloaders.datasets.utils import generate_gaussian_heatmaps
from dataloaders.transforms.spatial_transform import SpatialTransform
from dataloaders.datasets.image_datasets.unid_img_dataset import UnidImageDataset


class ImageCOCODataset(UnidImageDataset):
    def __init__(self, resize=(480, 640), crop_scale=(0.6, 1.0), crop_ratio=(1.0,1.0), near_plane=1e-5, far_plane=80.0, full=True, fill_missing_tasks=True, load_tasks=None, **kwargs):
        super().__init__(resize=resize, crop_scale=crop_scale, crop_ratio=crop_ratio, near_plane=near_plane, far_plane=far_plane, fill_missing_tasks=fill_missing_tasks, **kwargs)
        self.config = DATASET_CONFIG["coco"]
        self.dataset_name = "coco"
        self.gt_tasks = {'rgb', 'semseg-coco', 'boundary', 'keypoint-coco'}
        self.load_tasks = load_tasks if load_tasks is None else set(load_tasks)    # if None, load all available tasks
        
        if full:
            self.df = pd.read_parquet(self._file_io(self.config["parquet_path"]))
            self.imgid2anno = self._load_coco_annotations()
            logging.info(f"[ImageCOCODataset] found {len(self.df)} images.")
        else:
            self.imgid2anno = {}
            logging.info(f"[ImageCOCODataset] (partial).")
    
    def _is_gt_task(self, task_name):
        return task_name in self.gt_tasks
    
    def _is_load_task(self, task_name):
        assert self._is_gt_task(task_name), f"{task_name} not in gt tasks"
        return self.load_tasks is None or task_name in self.load_tasks

    def _load_coco_annotations(self):
        panoptic_anno_path = osp.join(
            self.config["prefix"], 'annotations', 'panoptic_train2017.json'
        )
        panoptic_anno = json.load(self._file_io(panoptic_anno_path))

        keypoint_anno_path = osp.join(
            self.config["prefix"], 'annotations', 'person_keypoints_train2017.json'
        )
        keypoint_anno = json.load(self._file_io(keypoint_anno_path))

        keypoint_imgid2anno = defaultdict(list)
        for anno in keypoint_anno['annotations']:
            keypoint_imgid2anno[anno['image_id']].append(anno)
        
        imgid2anno = {
            anno['image_id']: {
                'panoptic_anno': anno,
                'keypoint_anno': keypoint_imgid2anno.get(anno['image_id'], None)
                } for anno in panoptic_anno['annotations']
        }
        assert len(imgid2anno) == len(panoptic_anno['annotations']), "Not every valid image has annotation"
        return imgid2anno
    
    def _load_panoptic_coco(self, data):
        if not self._is_load_task('semseg-coco') and not self._is_load_task('boundary'):
            return None
        image_id = data['image_id']
        panoptic_anno = self.imgid2anno[image_id]['panoptic_anno']
        pan_mask_path = data['pan_anno_path']
        if pan_mask_path is not None and panoptic_anno is not None:
            pan_mask = rgb2id(np.array(Image.open(self._file_io(pan_mask_path))))
            pan_info = panoptic_anno['segments_info']
            [k.update({'isthing': self.sem_categories[k['category_id']]['isthing']}) for k in pan_info]
            return pan_mask, pan_info
        else:
            return None
    
    def _load_keypoint_coco(self, data):
        if not self._is_load_task('keypoint-coco'):
            return None
        image_id = data['image_id']
        keypoint_anno = self.imgid2anno[image_id]['keypoint_anno']
        if keypoint_anno is not None:
            keypoints = np.stack([
                np.array(anno['keypoints']).reshape(-1,3) for anno in keypoint_anno
            ], axis=0)  # N, 17, 3
            return keypoints
        else:
            return None

def generate_parquet_file(split="train", input_root=None, output_root=None):
    DATA_DIR = str(dataset_root("coco", input_root))

    img_dir = osp.join(DATA_DIR, f'{split}2017')
    img_paths = sorted(glob.glob(osp.join(img_dir, '*.jpg')))
    pseudo_depth_dir = osp.join(DATA_DIR, 'annotations', f'depth-pro_pred')
    pseudo_normal_dir = osp.join(DATA_DIR, 'annotations', f'dsine_pred')
    pan_anno_dir = osp.join(DATA_DIR, 'annotations', f'panoptic_{split}2017')
    require_files(
        osp.join(DATA_DIR, 'annotations', f'panoptic_{split}2017.json'),
        osp.join(DATA_DIR, 'annotations', f'person_keypoints_{split}2017.json'),
    )
    
    random.seed(1)
    data_list = []
    for image_path in tqdm.tqdm(img_paths, desc="Processing instances"):
        image_id = int(osp.basename(image_path).removesuffix('.jpg'))
        pseudo_depth_path = osp.join(pseudo_depth_dir, f'{image_id:012d}.png')
        pseudo_normal_path = osp.join(pseudo_normal_dir, f'{image_id:012d}.png')
        pan_anno_path = osp.join(pan_anno_dir, f'{image_id:012d}.png')
        require_files(pan_anno_path)

        frame_path = image_path
        if not osp.exists(pseudo_depth_path) or random.random() >= 0.1:
            pseudo_depth_path = None
        
        if not osp.exists(pseudo_normal_path) or random.random() >= 0.1:
            pseudo_normal_path = None

        data_list.append({
            'image_id': image_id,
            'rgb_path': frame_path,
            'pan_anno_path': pan_anno_path,
            'pseudo_depth_path': pseudo_depth_path,
            'pseudo_normal_path': pseudo_normal_path,
        })

    return write_index(data_list, "coco", split, output_root)


if __name__ == "__main__":
    # generate_parquet_file()

    # sample and visualize:
    print("\nTesting image dataset...")
    image_dataset = ImageCOCODataset()
    print(f"Image dataset length: {len(image_dataset)}")

    # image_sample = image_dataset[4]

    torch.manual_seed(1); rand_proj = torch.rand(3,17)
    instances = (torch.rand(10) * len(image_dataset)).long().tolist()
    for ii in tqdm.tqdm(instances):
        x = []
        for i in range(5):
            image_sample = image_dataset[ii]
            rgb = ((image_sample['rgb'] + 1)/2).permute(1,2,0).numpy()
            depth = image_sample['depth'].permute(1,2,0).numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = np.concatenate([depth]*3, axis=2)
            normal = ((image_sample['normal'] + 1)/2).permute(1,2,0).numpy()
            semseg = image_sample['semseg-coco']
            semantic_ids = torch.unique(semseg)
            semseg_vis = torch.zeros(rgb.shape)
            for sid in semantic_ids:
                semseg_vis[semseg==sid] = image_dataset.sem_color_code[sid.long()]
            semseg = semseg_vis.numpy()

            boundary = image_sample['boundary'].permute(1,2,0).numpy()
            boundary = np.concatenate([boundary]*3, axis=2)

            keypoint = rand_proj @ image_sample['keypoint-coco'].reshape(17, -1)  # 17, H, W
            keypoint = keypoint.reshape(3, *rgb.shape[:2]).permute(1,2,0).numpy()
            keypoint = (keypoint - keypoint.min()) / (keypoint.max() - keypoint.min() + 1e-8)

            out_img = np.hstack((rgb, depth, normal, semseg, boundary, keypoint)) * 255
            x.append(out_img)
        os.makedirs("data_vis", exist_ok=True)
        imageio.imwrite(f"data_vis/coco_sample_{ii}.png", np.vstack(x).astype(np.uint8))
