import os, cv2, json, random, tqdm
import os.path as osp
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataloaders.config import DATASET_CONFIG
from dataloaders.datasets.utils import generate_gaussian_heatmaps, panoptic2semseg, panoptic2boundary, get_pan_data
from dataloaders.transforms.spatial_transform import SpatialTransform
from dataloaders.transforms.blur_transform import BlurTransform

COCO_CATEGORIES = {
    0: {'name': 'void', 'supercategory': 'void'}, 1: {'name': 'person', 'supercategory': 'person', 'isthing': 1}, 2: {'name': 'bicycle', 'supercategory': 'vehicle', 'isthing': 1}, 3: {'name': 'car', 'supercategory': 'vehicle', 'isthing': 1}, 4: {'name': 'motorcycle', 'supercategory': 'vehicle', 'isthing': 1}, 5: {'name': 'airplane', 'supercategory': 'vehicle', 'isthing': 1}, 6: {'name': 'bus', 'supercategory': 'vehicle', 'isthing': 1}, 7: {'name': 'train', 'supercategory': 'vehicle', 'isthing': 1}, 8: {'name': 'truck', 'supercategory': 'vehicle', 'isthing': 1}, 9: {'name': 'boat', 'supercategory': 'vehicle', 'isthing': 1},
    10: {'name': 'traffic light', 'supercategory': 'outdoor', 'isthing': 1}, 11: {'name': 'fire hydrant', 'supercategory': 'outdoor', 'isthing': 1}, 13: {'name': 'stop sign', 'supercategory': 'outdoor', 'isthing': 1}, 14: {'name': 'parking meter', 'supercategory': 'outdoor', 'isthing': 1}, 15: {'name': 'bench', 'supercategory': 'outdoor', 'isthing': 1}, 16: {'name': 'bird', 'supercategory': 'animal', 'isthing': 1}, 17: {'name': 'cat', 'supercategory': 'animal', 'isthing': 1}, 18: {'name': 'dog', 'supercategory': 'animal', 'isthing': 1}, 19: {'name': 'horse', 'supercategory': 'animal', 'isthing': 1},
    20: {'name': 'sheep', 'supercategory': 'animal', 'isthing': 1}, 21: {'name': 'cow', 'supercategory': 'animal', 'isthing': 1}, 22: {'name': 'elephant', 'supercategory': 'animal', 'isthing': 1}, 23: {'name': 'bear', 'supercategory': 'animal', 'isthing': 1}, 24: {'name': 'zebra', 'supercategory': 'animal', 'isthing': 1}, 25: {'name': 'giraffe', 'supercategory': 'animal', 'isthing': 1}, 27: {'name': 'backpack', 'supercategory': 'accessory', 'isthing': 1}, 28: {'name': 'umbrella', 'supercategory': 'accessory', 'isthing': 1}, 31: {'name': 'handbag', 'supercategory': 'accessory', 'isthing': 1}, 32: {'name': 'tie', 'supercategory': 'accessory', 'isthing': 1}, 33: {'name': 'suitcase', 'supercategory': 'accessory', 'isthing': 1}, 34: {'name': 'frisbee', 'supercategory': 'sports', 'isthing': 1}, 35: {'name': 'skis', 'supercategory': 'sports', 'isthing': 1}, 36: {'name': 'snowboard', 'supercategory': 'sports', 'isthing': 1}, 37: {'name': 'sports ball', 'supercategory': 'sports', 'isthing': 1}, 38: {'name': 'kite', 'supercategory': 'sports', 'isthing': 1}, 39: {'name': 'baseball bat', 'supercategory': 'sports', 'isthing': 1},
    40: {'name': 'baseball glove', 'supercategory': 'sports', 'isthing': 1}, 41: {'name': 'skateboard', 'supercategory': 'sports', 'isthing': 1}, 42: {'name': 'surfboard', 'supercategory': 'sports', 'isthing': 1}, 43: {'name': 'tennis racket', 'supercategory': 'sports', 'isthing': 1}, 44: {'name': 'bottle', 'supercategory': 'kitchen', 'isthing': 1}, 46: {'name': 'wine glass', 'supercategory': 'kitchen', 'isthing': 1}, 47: {'name': 'cup', 'supercategory': 'kitchen', 'isthing': 1}, 48: {'name': 'fork', 'supercategory': 'kitchen', 'isthing': 1}, 49: {'name': 'knife', 'supercategory': 'kitchen', 'isthing': 1},
    50: {'name': 'spoon', 'supercategory': 'kitchen', 'isthing': 1}, 51: {'name': 'bowl', 'supercategory': 'kitchen', 'isthing': 1}, 52: {'name': 'banana', 'supercategory': 'food', 'isthing': 1}, 53: {'name': 'apple', 'supercategory': 'food', 'isthing': 1}, 54: {'name': 'sandwich', 'supercategory': 'food', 'isthing': 1}, 55: {'name': 'orange', 'supercategory': 'food', 'isthing': 1}, 56: {'name': 'broccoli', 'supercategory': 'food', 'isthing': 1}, 57: {'name': 'carrot', 'supercategory': 'food', 'isthing': 1}, 58: {'name': 'hot dog', 'supercategory': 'food', 'isthing': 1}, 59: {'name': 'pizza', 'supercategory': 'food', 'isthing': 1},
    60: {'name': 'donut', 'supercategory': 'food', 'isthing': 1}, 61: {'name': 'cake', 'supercategory': 'food', 'isthing': 1}, 62: {'name': 'chair', 'supercategory': 'furniture', 'isthing': 1}, 63: {'name': 'couch', 'supercategory': 'furniture', 'isthing': 1}, 64: {'name': 'potted plant', 'supercategory': 'furniture', 'isthing': 1}, 65: {'name': 'bed', 'supercategory': 'furniture', 'isthing': 1}, 67: {'name': 'dining table', 'supercategory': 'furniture', 'isthing': 1},
    70: {'name': 'toilet', 'supercategory': 'furniture', 'isthing': 1}, 72: {'name': 'tv', 'supercategory': 'electronic', 'isthing': 1}, 73: {'name': 'laptop', 'supercategory': 'electronic', 'isthing': 1}, 74: {'name': 'mouse', 'supercategory': 'electronic', 'isthing': 1}, 75: {'name': 'remote', 'supercategory': 'electronic', 'isthing': 1}, 76: {'name': 'keyboard', 'supercategory': 'electronic', 'isthing': 1}, 77: {'name': 'cell phone', 'supercategory': 'electronic', 'isthing': 1}, 78: {'name': 'microwave', 'supercategory': 'appliance', 'isthing': 1}, 79: {'name': 'oven', 'supercategory': 'appliance', 'isthing': 1},
    80: {'name': 'toaster', 'supercategory': 'appliance', 'isthing': 1}, 81: {'name': 'sink', 'supercategory': 'appliance', 'isthing': 1}, 82: {'name': 'refrigerator', 'supercategory': 'appliance', 'isthing': 1}, 84: {'name': 'book', 'supercategory': 'indoor', 'isthing': 1}, 85: {'name': 'clock', 'supercategory': 'indoor', 'isthing': 1}, 86: {'name': 'vase', 'supercategory': 'indoor', 'isthing': 1}, 87: {'name': 'scissors', 'supercategory': 'indoor', 'isthing': 1}, 88: {'name': 'teddy bear', 'supercategory': 'indoor', 'isthing': 1}, 89: {'name': 'hair drier', 'supercategory': 'indoor', 'isthing': 1},
    90: {'name': 'toothbrush', 'supercategory': 'indoor', 'isthing': 1}, 92: {'name': 'banner', 'supercategory': 'textile', 'isthing': 0}, 93: {'name': 'blanket', 'supercategory': 'textile', 'isthing': 0}, 95: {'name': 'bridge', 'supercategory': 'building', 'isthing': 0},
    100: {'name': 'cardboard', 'supercategory': 'raw-material', 'isthing': 0}, 107: {'name': 'counter', 'supercategory': 'furniture-stuff', 'isthing': 0}, 109: {'name': 'curtain', 'supercategory': 'textile', 'isthing': 0}, 112: {'name': 'door-stuff', 'supercategory': 'furniture-stuff', 'isthing': 0}, 118: {'name': 'floor-wood', 'supercategory': 'floor', 'isthing': 0}, 119: {'name': 'flower', 'supercategory': 'plant', 'isthing': 0}, 122: {'name': 'fruit', 'supercategory': 'food-stuff', 'isthing': 0}, 125: {'name': 'gravel', 'supercategory': 'ground', 'isthing': 0}, 128: {'name': 'house', 'supercategory': 'building', 'isthing': 0},
    130: {'name': 'light', 'supercategory': 'furniture-stuff', 'isthing': 0}, 133: {'name': 'mirror-stuff', 'supercategory': 'furniture-stuff', 'isthing': 0}, 138: {'name': 'net', 'supercategory': 'structural', 'isthing': 0}, 141: {'name': 'pillow', 'supercategory': 'textile', 'isthing': 0}, 144: {'name': 'platform', 'supercategory': 'ground', 'isthing': 0}, 145: {'name': 'playingfield', 'supercategory': 'ground', 'isthing': 0}, 147: {'name': 'railroad', 'supercategory': 'ground', 'isthing': 0}, 148: {'name': 'river', 'supercategory': 'water', 'isthing': 0}, 149: {'name': 'road', 'supercategory': 'ground', 'isthing': 0}, 151: {'name': 'roof', 'supercategory': 'building', 'isthing': 0}, 154: {'name': 'sand', 'supercategory': 'ground', 'isthing': 0}, 155: {'name': 'sea', 'supercategory': 'water', 'isthing': 0}, 156: {'name': 'shelf', 'supercategory': 'furniture-stuff', 'isthing': 0}, 159: {'name': 'snow', 'supercategory': 'ground', 'isthing': 0}, 161: {'name': 'stairs', 'supercategory': 'furniture-stuff', 'isthing': 0}, 166: {'name': 'tent', 'supercategory': 'building', 'isthing': 0}, 168: {'name': 'towel', 'supercategory': 'textile', 'isthing': 0}, 171: {'name': 'wall-brick', 'supercategory': 'wall', 'isthing': 0}, 175: {'name': 'wall-stone', 'supercategory': 'wall', 'isthing': 0}, 176: {'name': 'wall-tile', 'supercategory': 'wall', 'isthing': 0}, 177: {'name': 'wall-wood', 'supercategory': 'wall', 'isthing': 0}, 178: {'name': 'water-other', 'supercategory': 'water', 'isthing': 0},
    180: {'name': 'window-blind', 'supercategory': 'window', 'isthing': 0}, 181: {'name': 'window-other', 'supercategory': 'window', 'isthing': 0}, 184: {'name': 'tree-merged', 'supercategory': 'plant', 'isthing': 0}, 185: {'name': 'fence-merged', 'supercategory': 'structural', 'isthing': 0}, 186: {'name': 'ceiling-merged', 'supercategory': 'ceiling', 'isthing': 0}, 187: {'name': 'sky-other-merged', 'supercategory': 'sky', 'isthing': 0}, 188: {'name': 'cabinet-merged', 'supercategory': 'furniture-stuff', 'isthing': 0}, 189: {'name': 'table-merged', 'supercategory': 'furniture-stuff', 'isthing': 0},
    190: {'name': 'floor-other-merged', 'supercategory': 'floor', 'isthing': 0}, 191: {'name': 'pavement-merged', 'supercategory': 'ground', 'isthing': 0}, 192: {'name': 'mountain-merged', 'supercategory': 'solid', 'isthing': 0}, 193: {'name': 'grass-merged', 'supercategory': 'plant', 'isthing': 0}, 194: {'name': 'dirt-merged', 'supercategory': 'ground', 'isthing': 0}, 195: {'name': 'paper-merged', 'supercategory': 'raw-material', 'isthing': 0}, 196: {'name': 'food-other-merged', 'supercategory': 'food-stuff', 'isthing': 0}, 197: {'name': 'building-other-merged', 'supercategory': 'building', 'isthing': 0}, 198: {'name': 'rock-merged', 'supercategory': 'solid', 'isthing': 0}, 199: {'name': 'wall-other-merged', 'supercategory': 'wall', 'isthing': 0},200: {'name': 'rug-merged', 'supercategory': 'textile', 'isthing': 0}
}


class UnidImageDataset(Dataset):
    def __init__(self, resize=(480, 640), crop_scale=(0.8, 1.0), crop_ratio=(1.0,1.0), blur_motion_kernel=(25,50), blur_kernel=(3,10), blur_prob=0, near_plane=1e-5, far_plane=65.0, fill_missing_tasks=True):
        self.resize = resize
        self.near_plane = near_plane
        self.far_plane  = far_plane
        r = resize[1] / resize[0]
        self.transform = SpatialTransform(
            size=resize,
            prob_flip=0.5,
            crop_scale=crop_scale,
            crop_ratio=(crop_ratio[0]*r, crop_ratio[1]*r)
        )
        self.blur_transform = BlurTransform(
            motion_kernel_range=blur_motion_kernel,
            blur_kernel_range=blur_kernel,
            angle_range=(0, 360),
            p=blur_prob
        )
        self.to_tensor = transforms.ToTensor()
        self.fill_missing_tasks = fill_missing_tasks

        # can be overridden by child classes
        self.sem_categories = self._get_sem_categories()
        self.sem_color_code = self._construct_color_code()

    def _get_sem_categories(self):
        """Organize frames by video name and sort by frame number."""
        categories = COCO_CATEGORIES    # no need to load full json file, use predefined dict (can be obtained from above code snippet)
        return categories

    def _construct_color_code(self, color_code_str='spacing'):
        max_cat = max(self.sem_categories.keys()) + 1
        color_code = torch.zeros((max_cat,3), dtype=torch.float32)

        if color_code_str == 'spacing':
            base = int(np.ceil(len(self.sem_categories) ** (1/3)))
            c_list = torch.linspace(0, 1, base)
            for k, (ii, v) in enumerate(self.sem_categories.items()):
                i = k // (base * base)
                j = (k // base) % base
                l = k % base
                color_code[ii] = torch.tensor([c_list[i], c_list[j], c_list[l]], dtype=torch.float32)
        else:
            raise NotImplementedError
        return color_code  # num_classes, 3

    def _file_io(self, file_path, bucket=None):
        """Open a locally supplied dataset or index path."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Required local dataset file is missing: {file_path}. Obtain the dataset "
                "from its upstream source and set UNID_DATA_ROOT to its local root."
            )
        return open(file_path, "rb")

    def __len__(self):
        return len(self.df)
    
    def load_data(self, data):
        rgb_image = self._load_rgb(data)
        depth_image = self._load_depth(data)
        normal_image = self._load_normal(data, depth_image)
        pan_data_coco = self._load_panoptic_coco(data)
        semseg_image_coco = panoptic2semseg(pan_data_coco)
        pan_data_cityscapes = self._load_panoptic_cityscapes(data)
        semseg_image_cityscapes = panoptic2semseg(pan_data_cityscapes)
        pan_data_vipseg = self._load_panoptic_vipseg(data)
        semseg_image_vipseg = panoptic2semseg(pan_data_vipseg)
        boundary_data = panoptic2boundary(get_pan_data([pan_data_coco, pan_data_cityscapes, pan_data_vipseg]))
        # keypoint_data = self._load_keypoint_coco(data)
        partseg_image = self._load_part_coco(data)
        material_image = self._load_material(data)
        albedo_image = self._load_albedo(data)
        illum_image = self._load_illum(data)

        return {
            'rgb': rgb_image,
            'depth': depth_image,
            'normal': normal_image,
            'boundary': boundary_data,
            # 'keypoint-coco': keypoint_data,
            'semseg-coco': semseg_image_coco,
            'semseg-cityscapes': semseg_image_cityscapes,
            'semseg-vipseg': semseg_image_vipseg,
            'matseg': material_image,
            'albedo': albedo_image,
            'illum': illum_image,
            'partseg-coco': partseg_image
        }
    
    def prepare_output(self, transformed):
        output, data_sup = dict(), dict()
        for k, data in transformed.items():
            if data is None and not self.fill_missing_tasks:
                continue
            task = k.split('-')[0]
            func = getattr(self, f'_process_{task}')
            out, presence = func(data)
            if torch.is_tensor(out):
                output[k] = out
            else:
                output.update(out)
            if self._is_gt_task(k):
                data_sup[k] = 2  # full supervision
            elif presence:
                data_sup[k] = 1  # pseudo supervision
            else:
                data_sup[k] = 0  # no supervision
        
        # handle multi-task dependencies
        if 'normal' in data_sup:
            if data_sup['normal'] > 0:
                if data_sup['depth'] > 0:
                    valid_mask = output['depth_valid_mask']
                else:
                    valid_mask = torch.ones_like(output['depth_valid_mask'], dtype=torch.bool)
                output['normal'][~valid_mask.repeat(3,1,1)] = 0.0
            else:
                valid_mask = torch.zeros_like(output['depth_valid_mask'], dtype=torch.bool)
            output['normal_valid_mask'] = valid_mask
        
        return output, data_sup

    def __getitem__(self, idx=None):
        if idx is None:
            idx = np.random.randint(len(self.df))

        data = self.df.iloc[idx]
        transformed = self.transform(self.load_data(data))
        transformed = self.blur_transform(transformed)
        output, data_sup = self.prepare_output(transformed)

        output.update({
            'supervision': data_sup,
            'dataset': self.dataset_name,
            'instance': idx,
        })

        return output


    #=== Data loading functions (can be overwritten by child classes) ===#

    def _is_load_task(self, task_name):
        return True     # can be overridden by child classes

    def _load_rgb(self, data):
        ''' Loading RGB image from path
        '''
        rgb_image = Image.open(self._file_io(data['rgb_path'])).convert('RGB')
        return rgb_image

    # def _load_depth(self, data):
    #     ''' Loading pseudo depth data (depth-pro)
    #     '''
    #     if not self._is_load_task('depth'):
    #         return None
    #     path = data['pseudo_depth_path'] if 'pseudo_depth_path' in data else None
    #     if path is not None:
    #         depth_image = Image.open(self._file_io(path))
    #         depth_image = np.array(depth_image, dtype=float) / 65535.0 * 1000  # storage
    #         depth_image = Image.fromarray(depth_image)
    #         return depth_image
    #     else:
    #         return None

    # def _load_normal(self, data, depth_image, align_cam_normal=True):
    #     ''' Loading pseudo normal data (DSINE)
    #     '''
    #     if not self._is_load_task('normal'):
    #         return None
    #     path = data['pseudo_normal_path'] if 'pseudo_normal_path' in data else None
    #     if path is not None:
    #         normal_image = Image.open(self._file_io(path))
    #         return normal_image
    #     else:
    #         return None

    # def _load_panoptic_coco(self, data):
    #     ''' Loading pseudo panoptic data from coco (OneFormer)
    #     '''
    #     if not self._is_load_task('semseg'):
    #         return None
    #     path = data['pseudo_coco_panoptic_path'] if 'pseudo_coco_panoptic_path' in data else None
    #     if path is not None:
    #         pan_mask = np.array(Image.open(self._file_io(path)))
    #         pan_info = json.load(self._file_io(path.replace('_panoptic.png', '_panoptic.json')))
    #         return pan_mask, pan_info
    #     else:
    #         return None

    # def _load_keypoint_coco(self, data):
    #     ''' Loading pseudo keypoints from coco (RTMO)
    #     '''
    #     if not self._is_load_task('keypoint'):
    #         return None
    #     path = data['pseudo_coco_keypoint_path'] if 'pseudo_coco_keypoint_path' in data else None
    #     if path is not None:
    #         kp_data = np.load(self._file_io(path))
    #         keypoints = kp_data['keypoints'].astype(float)
    #         scores = (kp_data['scores'].astype(float) > 0.5).astype(float)
    #         return np.concatenate([keypoints, scores[:,:,None]], axis=2).tolist()
    #     else:
    #         return None
    
    def _load_depth(self, data):
        return None # to be implemented in child dataset class
    
    def _load_normal(self, data, depth_image, align_cam_normal=True):
        return None # to be implemented in child dataset class
    
    def _load_panoptic_coco(self, data):
        return None # to be implemented in child dataset class
    
    def _load_panoptic_cityscapes(self, data):
        return None # to be implemented in child Cityscapes dataset class
    
    def _load_panoptic_vipseg(self, data):
        return None # to be implemented in child VIPSeg dataset class
    
    def _load_material(self, data):
        return None # to be implemented in child DMS dataset class

    def _load_albedo(self, data):
        return None # to be implemented in child dataset class

    def _load_illum(self, data):
        return None # to be implemented in child dataset class
    
    def _load_part_coco(self, data):
        return None # to be implemented in child dataset class


    #=== Data processing functions (can be overwritten by child classes) ===#

    def _is_gt_task(self, task_name):
        pass

    def _process_rgb(self, rgb_image):
        ''' Process RGB image to ensure proper scaling
            Output: 3, H, W tensor in [-1, 1]
        '''
        return self.to_tensor(rgb_image) * 2 - 1, True

    def _process_depth(self, depth_image):
        ''' Process depth image to ensure proper scaling and handle invalid values
            Output: 1, H, W tensor in meters
        '''
        if depth_image is None:
            metric_tensor = torch.zeros((1, *self.resize), dtype=torch.float32)
            valid_depth_mask = torch.zeros((1, *self.resize), dtype=torch.bool)
            far_invalid_depth_mask = torch.zeros((1, *self.resize), dtype=torch.bool)
            found = False
        else:
            depth_tensor = self.to_tensor(depth_image)
            valid_depth_mask = (depth_tensor > self.near_plane) & (depth_tensor < self.far_plane)
            far_invalid_depth_mask = depth_tensor >= self.far_plane
            found = True

            if valid_depth_mask.any():
                flat_depth = depth_tensor[valid_depth_mask].flatten().float()
                min_depth = flat_depth.min()
                max_depth = flat_depth.max()
                if min_depth == max_depth:
                    metric_tensor = torch.zeros_like(depth_tensor)
                    valid_depth_mask = torch.zeros_like(depth_tensor).bool() # empty mask
                    far_invalid_depth_mask = torch.zeros_like(depth_tensor).bool() # empty mask
                else:
                    depth_tensor = torch.clamp(depth_tensor, min_depth, max_depth) # remove outliers
                    depth_tensor[~valid_depth_mask] = max_depth                    # set invalid depth to relative far plane
                    metric_tensor = depth_tensor.clone()                           # keep metric depth for e2e loss ft
            else:
                metric_tensor = torch.zeros_like(depth_tensor)
        
        out = {
            'depth': metric_tensor,
            'depth_valid_mask': valid_depth_mask,
            'depth_invalid_far_mask': far_invalid_depth_mask
        }
        return out, found

    def _process_normal(self, normal_image):
        ''' Process normal image to ensure proper scaling and handle invalid values
            Output: 3, H, W tensor in [-1, 1]
        '''
        if normal_image is None:
            placeholder = torch.zeros((3, *self.resize), dtype=torch.float32)
            placeholder[2,:,:] = -1.0  # invalid normal pointing backward (placeholder)
            return placeholder, False
        return self.to_tensor(normal_image) * 2 - 1, True

    def _process_semseg(self, semseg_image):
        ''' Process semantic segmentation image to ensure proper format
            Output: H, W tensor in uint8
        '''
        if semseg_image is None:
            return torch.zeros(self.resize, dtype=torch.uint8), False
        return torch.from_numpy(np.array(semseg_image)).long(), True
    
    def _process_boundary(self, boundary_data):
        ''' Process boundary data to ensure proper format
            Output: 1, H, W tensor in float32
        '''
        if boundary_data is None:
            boundary = torch.zeros((1, *self.resize), dtype=torch.float32)
            valid = torch.zeros((1, *self.resize), dtype=torch.bool)
            found = False
        else:
            boundary_contours = boundary_data
            boundary_image = np.zeros(self.resize, dtype=np.uint8)
            cv2.drawContours(boundary_image, [np.array(contour).reshape(-1,1,2).astype(np.int32) for contour in boundary_contours if len(contour)>=6], -1, color=1, thickness=1)
            boundary = torch.from_numpy(boundary_image).unsqueeze(0).float()
            valid = torch.ones((1, *self.resize), dtype=torch.bool)
            found = True
        return {'boundary': boundary, 'boundary_valid_mask': valid}, found
    
    def _process_input(self, input_image):
        ''' Process inputd image to ensure proper scaling
            Output: 3, H, W tensor in [-1, 1]
        '''
        return self.to_tensor(input_image) * 2 - 1, True
        
    # def _process_keypoint(self, keypoint_data):
    #     ''' Process keypoint data to ensure proper format
    #         Output: 17, H, W tensor in float32
    #     '''
    #     if keypoint_data is None:
    #         return torch.zeros((17, *self.resize), dtype=torch.float32), False
    #     kp_map =  generate_gaussian_heatmaps(
    #         keypoint_data,
    #         image_shape=self.resize,
    #         sigma=1.0,
    #         kernel_size=7,
    #         num_keypoints=17
    #     )
    #     return kp_map, True

    def _process_matseg(self, semseg_image):
        ''' Process semantic segmentation image to ensure proper format
            Output: H, W tensor in uint8
        '''
        return self._process_semseg(semseg_image)

    def _process_partseg(self, semseg_image):
        ''' Process semantic segmentation image to ensure proper format
            Output: H, W tensor in uint8
        '''
        return self._process_semseg(semseg_image)

    def _process_albedo(self, albedo_image):
        ''' Process semantic segmentation image to ensure proper format
            Output: H, W tensor in uint8
        '''
        if albedo_image is None:
            return torch.zeros(self.resize, dtype=torch.uint8), False
        return self.to_tensor(albedo_image) * 2 - 1, True

    def _process_illum(self, illum_image):
        ''' Process semantic segmentation image to ensure proper format
            Output: H, W tensor in uint8
        '''
        if illum_image is None:
            return torch.zeros(self.resize, dtype=torch.uint8), False
        return self.to_tensor(illum_image) * 2 - 1, True
