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

CITYSCAPES_CATEGORIES = {
    #      name                       trainId   category            catId     hasInstances   ignoreInEval   color
     0: (  'unlabeled'            ,       255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
     1: (  'ego vehicle'          ,       255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
     2: (  'rectification border' ,       255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
     3: (  'out of roi'           ,       255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
     4: (  'static'               ,       255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
     5: (  'dynamic'              ,       255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
     6: (  'ground'               ,       255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
     7: (  'road'                 ,         0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
     8: (  'sidewalk'             ,         1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
     9: (  'parking'              ,       255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    10: (  'rail track'           ,       255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    11: (  'building'             ,         2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    12: (  'wall'                 ,         3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    13: (  'fence'                ,         4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    14: (  'guard rail'           ,       255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    15: (  'bridge'               ,       255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    16: (  'tunnel'               ,       255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    17: (  'pole'                 ,         5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    18: (  'polegroup'            ,       255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    19: (  'traffic light'        ,         6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    20: (  'traffic sign'         ,         7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    21: (  'vegetation'           ,         8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    22: (  'terrain'              ,         9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    23: (  'sky'                  ,        10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    24: (  'person'               ,        11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    25: (  'rider'                ,        12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    26: (  'car'                  ,        13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    27: (  'truck'                ,        14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    28: (  'bus'                  ,        15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    29: (  'caravan'              ,       255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    30: (  'trailer'              ,       255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    31: (  'train'                ,        16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    32: (  'motorcycle'           ,        17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    33: (  'bicycle'              ,        18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    34: (  'license plate'        ,        -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
}


class ImageCityscapesDataset(UnidImageDataset):
    def __init__(self, resize=(480, 640), crop_scale=(0.6, 1.0), crop_ratio=(1.0,1.0), near_plane=1e-5, far_plane=80.0, full=True, fill_missing_tasks=True, load_tasks=None, **kwargs):
        super().__init__(resize=resize, crop_scale=crop_scale, crop_ratio=crop_ratio, near_plane=near_plane, far_plane=far_plane, fill_missing_tasks=fill_missing_tasks, **kwargs)
        self.config = DATASET_CONFIG["cityscapes"]
        self.dataset_name = "cityscapes"
        self.gt_tasks = {'rgb', 'semseg-cityscapes', 'boundary'}
        self.sem_categories = self._get_sem_categories()
        self.sem_color_code = self._construct_color_code()
        self.load_tasks = load_tasks if load_tasks is None else set(load_tasks)    # if None, load all available tasks

        if full:
            self.df = pd.read_parquet(self._file_io(self.config["parquet_path"]))
            logging.info(f"[Cityscapes] found {len(self.df)} images.")

    def _get_sem_categories(self):
        """Organize frames by video name and sort by frame number."""
        categories = {
            k: {'name': v[0], 'supercategory': v[2], 'isthing': v[4]} for k, v in CITYSCAPES_CATEGORIES.items()
        }
        return categories

    def _construct_color_code(self):
        color_code = torch.zeros((max(CITYSCAPES_CATEGORIES.keys())+1,3), dtype=torch.float32)
        for k, v in CITYSCAPES_CATEGORIES.items():
            color_code[k] = torch.tensor(CITYSCAPES_CATEGORIES[k][6], dtype=torch.float32) / 255.0
        return color_code  # num_classes, 3
    
    def _is_gt_task(self, task_name):
        return task_name in self.gt_tasks
    
    def _is_load_task(self, task_name):
        assert self._is_gt_task(task_name), f"{task_name} not in gt tasks"
        return self.load_tasks is None or task_name in self.load_tasks
    
    def _load_panoptic_cityscapes(self, data):
        if not self._is_load_task('semseg-cityscapes') and not self._is_load_task('boundary'):
            return None
        pan_mask = np.array(Image.open(self._file_io(data['pan_anno_path'])))
        instance_ids = np.unique(pan_mask)
        pan_info = []
        for inst_id in instance_ids:
            if inst_id == 0:
                continue  # skip background
            semantic_id = inst_id if inst_id < 1000 else inst_id // 1000
            isthing = self.sem_categories[semantic_id]['isthing']
            pan_info.append({
                'id': int(inst_id),
                'category_id': int(semantic_id),
                'isthing': isthing,
            })

        return pan_mask, pan_info

def generate_parquet_file(split="train", input_root=None, output_root=None):
    DATA_DIR = str(dataset_root("cityscapes", input_root))

    data_list = []
    locations = sorted(os.listdir(osp.join(DATA_DIR, 'leftImg8bit', split)))
    for loc in locations:
        frame_paths = sorted(os.listdir(osp.join(DATA_DIR, 'leftImg8bit', split, loc)))
        for frame_path in frame_paths:
            image_name = osp.basename(osp.splitext(frame_path)[0]).removesuffix('_leftImg8bit')
            rgb_path = osp.join(DATA_DIR, 'leftImg8bit', split, loc, frame_path)
            anno_path = osp.join(DATA_DIR, 'gtFine', split, loc, image_name+'_gtFine_instanceIds.png')
            require_files(rgb_path, anno_path)

            pan_anno_path = anno_path

            pseudo_depth_path = osp.join(DATA_DIR, split+'_depth-pro_pred', loc, image_name+'_leftImg8bit_depth.png')
            pseudo_normal_path = osp.join(DATA_DIR, split+'_dsine_pred', loc, image_name+'_leftImg8bit_normal.png')
            pseudo_kp_path = osp.join(DATA_DIR, split+'_rtmo_pred', loc, image_name+'_leftImg8bit_keypoint.npz')

            if not osp.exists(pseudo_depth_path):
                pseudo_depth_path = None
            
            if not osp.exists(pseudo_normal_path):
                pseudo_normal_path = None
            
            if not osp.exists(pseudo_kp_path):
                pseudo_kp_path = None
            
            data_list.append({
                'rgb_path': rgb_path,
                'pseudo_depth_path': pseudo_depth_path,
                'pseudo_normal_path': pseudo_normal_path,
                'pan_anno_path': pan_anno_path,
                'pseudo_coco_panoptic_path': None, # never pseudo label since we have GT
                'pseudo_coco_keypoint_path': pseudo_kp_path,
            })

    return write_index(data_list, "cityscapes", split, output_root)


if __name__ == "__main__":
    generate_parquet_file()

    # sample and visualize:
    print("\nTesting image dataset...")
    image_dataset = ImageCityscapesDataset()
    print(f"Image dataset length: {len(image_dataset)}")

    # image_sample = image_dataset[4]

    torch.manual_seed(1); rand_proj = torch.rand(3,17)
    instances = (torch.rand(10) * len(image_dataset)).long().tolist()
    instances = [0, 1, 2, 3, 4]
    for ii in tqdm.tqdm(instances):
        x = []
        for i in range(5):
            image_sample = image_dataset[ii]
            rgb = ((image_sample['rgb'] + 1)/2).permute(1,2,0).numpy()
            depth = image_sample['depth'].permute(1,2,0).numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = np.concatenate([depth]*3, axis=2)
            normal = ((image_sample['normal'] + 1)/2).permute(1,2,0).numpy()
            semseg = image_sample['semseg-cityscapes']
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
        imageio.imwrite(f"data_vis/cityscapes_sample_{ii}.png", np.vstack(x).astype(np.uint8))
