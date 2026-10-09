import os, cv2, io, json, random, logging, tqdm, glob
import os.path as osp
import pandas as pd
import numpy as np
import h5py
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataloaders.config import DATASET_CONFIG
from dataloaders.preparation import dataset_root, require_files, write_index
from dataloaders.datasets.utils import generate_gaussian_heatmaps
from dataloaders.transforms.spatial_transform import SpatialTransform
from dataloaders.datasets.video_datasets.unid_vid_dataset import UnidVideoDataset

VIPSEG_CATEGORIES = {
    0: {'name': 'invalid', 'isthing': 0, 'color': [0, 0, 0]},
    1: {'name': 'wall', 'isthing': 0, 'color': [120, 120, 120]},
    2: {'name': 'ceiling', 'isthing': 0, 'color': [180, 120, 120]},
    3: {'name': 'door', 'isthing': 1, 'color': [6, 230, 230]},
    4: {'name': 'stair', 'isthing': 0, 'color': [80, 50, 50]},
    5: {'name': 'ladder', 'isthing': 1, 'color': [4, 200, 3]},
    6: {'name': 'escalator', 'isthing': 0, 'color': [120, 120, 80]},
    7: {'name': 'Playground_slide', 'isthing': 0, 'color': [140, 140, 140]},
    8: {'name': 'handrail_or_fence', 'isthing': 0, 'color': [204, 5, 255]},
    9: {'name': 'window', 'isthing': 1, 'color': [230, 230, 230]},
    10: {'name': 'rail', 'isthing': 0, 'color': [4, 250, 7]},
    11: {'name': 'goal', 'isthing': 1, 'color': [224, 5, 255]},
    12: {'name': 'pillar', 'isthing': 0, 'color': [235, 255, 7]},
    13: {'name': 'pole', 'isthing': 0, 'color': [150, 5, 61]},
    14: {'name': 'floor', 'isthing': 0, 'color': [120, 120, 70]},
    15: {'name': 'ground', 'isthing': 0, 'color': [8, 255, 51]},
    16: {'name': 'grass', 'isthing': 0, 'color': [255, 6, 82]},
    17: {'name': 'sand', 'isthing': 0, 'color': [143, 255, 140]},
    18: {'name': 'athletic_field', 'isthing': 0, 'color': [204, 255, 4]},
    19: {'name': 'road', 'isthing': 0, 'color': [255, 51, 7]},
    20: {'name': 'path', 'isthing': 0, 'color': [204, 70, 3]},
    21: {'name': 'crosswalk', 'isthing': 0, 'color': [0, 102, 200]},
    22: {'name': 'building', 'isthing': 0, 'color': [61, 230, 250]},
    23: {'name': 'house', 'isthing': 0, 'color': [255, 6, 51]},
    24: {'name': 'bridge', 'isthing': 0, 'color': [11, 102, 255]},
    25: {'name': 'tower', 'isthing': 0, 'color': [255, 7, 71]},
    26: {'name': 'windmill', 'isthing': 0, 'color': [255, 9, 224]},
    27: {'name': 'well_or_well_lid', 'isthing': 0, 'color': [9, 7, 230]},
    28: {'name': 'other_construction', 'isthing': 0, 'color': [220, 220, 220]},
    29: {'name': 'sky', 'isthing': 0, 'color': [255, 9, 92]},
    30: {'name': 'mountain', 'isthing': 0, 'color': [112, 9, 255]},
    31: {'name': 'stone', 'isthing': 0, 'color': [8, 255, 214]},
    32: {'name': 'wood', 'isthing': 0, 'color': [7, 255, 224]},
    33: {'name': 'ice', 'isthing': 0, 'color': [255, 184, 6]},
    34: {'name': 'snowfield', 'isthing': 0, 'color': [10, 255, 71]},
    35: {'name': 'grandstand', 'isthing': 0, 'color': [255, 41, 10]},
    36: {'name': 'sea', 'isthing': 0, 'color': [7, 255, 255]},
    37: {'name': 'river', 'isthing': 0, 'color': [224, 255, 8]},
    38: {'name': 'lake', 'isthing': 0, 'color': [102, 8, 255]},
    39: {'name': 'waterfall', 'isthing': 0, 'color': [255, 61, 6]},
    40: {'name': 'water', 'isthing': 0, 'color': [255, 194, 7]},
    41: {'name': 'billboard_or_Bulletin_Board', 'isthing': 0, 'color': [255, 122, 8]},
    42: {'name': 'sculpture', 'isthing': 1, 'color': [0, 255, 20]},
    43: {'name': 'pipeline', 'isthing': 0, 'color': [255, 8, 41]},
    44: {'name': 'flag', 'isthing': 1, 'color': [255, 5, 153]},
    45: {'name': 'parasol_or_umbrella', 'isthing': 1, 'color': [6, 51, 255]},
    46: {'name': 'cushion_or_carpet', 'isthing': 0, 'color': [235, 12, 255]},
    47: {'name': 'tent', 'isthing': 1, 'color': [160, 150, 20]},
    48: {'name': 'roadblock', 'isthing': 1, 'color': [0, 163, 255]},
    49: {'name': 'car', 'isthing': 1, 'color': [140, 140, 140]},
    50: {'name': 'bus', 'isthing': 1, 'color': [250, 10, 15]},
    51: {'name': 'truck', 'isthing': 1, 'color': [20, 255, 0]},
    52: {'name': 'bicycle', 'isthing': 1, 'color': [31, 255, 0]},
    53: {'name': 'motorcycle', 'isthing': 1, 'color': [255, 31, 0]},
    54: {'name': 'wheeled_machine', 'isthing': 0, 'color': [255, 224, 0]},
    55: {'name': 'ship_or_boat', 'isthing': 1, 'color': [153, 255, 0]},
    56: {'name': 'raft', 'isthing': 1, 'color': [0, 0, 255]},
    57: {'name': 'airplane', 'isthing': 1, 'color': [255, 71, 0]},
    58: {'name': 'tyre', 'isthing': 0, 'color': [0, 235, 255]},
    59: {'name': 'traffic_light', 'isthing': 0, 'color': [0, 173, 255]},
    60: {'name': 'lamp', 'isthing': 0, 'color': [31, 0, 255]},
    61: {'name': 'person', 'isthing': 1, 'color': [11, 200, 200]},
    62: {'name': 'cat', 'isthing': 1, 'color': [255, 82, 0]},
    63: {'name': 'dog', 'isthing': 1, 'color': [0, 255, 245]},
    64: {'name': 'horse', 'isthing': 1, 'color': [0, 61, 255]},
    65: {'name': 'cattle', 'isthing': 1, 'color': [0, 255, 112]},
    66: {'name': 'other_animal', 'isthing': 1, 'color': [0, 255, 133]},
    67: {'name': 'tree', 'isthing': 0, 'color': [255, 0, 0]},
    68: {'name': 'flower', 'isthing': 0, 'color': [255, 163, 0]},
    69: {'name': 'other_plant', 'isthing': 0, 'color': [255, 102, 0]},
    70: {'name': 'toy', 'isthing': 0, 'color': [194, 255, 0]},
    71: {'name': 'ball_net', 'isthing': 0, 'color': [0, 143, 255]},
    72: {'name': 'backboard', 'isthing': 0, 'color': [51, 255, 0]},
    73: {'name': 'skateboard', 'isthing': 1, 'color': [0, 82, 255]},
    74: {'name': 'bat', 'isthing': 0, 'color': [0, 255, 41]},
    75: {'name': 'ball', 'isthing': 1, 'color': [0, 255, 173]},
    76: {'name': 'cupboard_or_showcase_or_storage_rack', 'isthing': 0, 'color': [10, 0, 255]},
    77: {'name': 'box', 'isthing': 1, 'color': [173, 255, 0]},
    78: {'name': 'traveling_case_or_trolley_case', 'isthing': 1, 'color': [0, 255, 153]},
    79: {'name': 'basket', 'isthing': 1, 'color': [255, 92, 0]},
    80: {'name': 'bag_or_package', 'isthing': 1, 'color': [255, 0, 255]},
    81: {'name': 'trash_can', 'isthing': 0, 'color': [255, 0, 245]},
    82: {'name': 'cage', 'isthing': 0, 'color': [255, 0, 102]},
    83: {'name': 'plate', 'isthing': 1, 'color': [255, 173, 0]},
    84: {'name': 'tub_or_bowl_or_pot', 'isthing': 1, 'color': [255, 0, 20]},
    85: {'name': 'bottle_or_cup', 'isthing': 1, 'color': [255, 184, 184]},
    86: {'name': 'barrel', 'isthing': 1, 'color': [0, 31, 255]},
    87: {'name': 'fishbowl', 'isthing': 1, 'color': [0, 255, 61]},
    88: {'name': 'bed', 'isthing': 1, 'color': [0, 71, 255]},
    89: {'name': 'pillow', 'isthing': 1, 'color': [255, 0, 204]},
    90: {'name': 'table_or_desk', 'isthing': 1, 'color': [0, 255, 194]},
    91: {'name': 'chair_or_seat', 'isthing': 1, 'color': [0, 255, 82]},
    92: {'name': 'bench', 'isthing': 1, 'color': [0, 10, 255]},
    93: {'name': 'sofa', 'isthing': 1, 'color': [0, 112, 255]},
    94: {'name': 'shelf', 'isthing': 0, 'color': [51, 0, 255]},
    95: {'name': 'bathtub', 'isthing': 0, 'color': [0, 194, 255]},
    96: {'name': 'gun', 'isthing': 1, 'color': [0, 122, 255]},
    97: {'name': 'commode', 'isthing': 1, 'color': [0, 255, 163]},
    98: {'name': 'roaster', 'isthing': 1, 'color': [255, 153, 0]},
    99: {'name': 'other_machine', 'isthing': 0, 'color': [0, 255, 10]},
    100: {'name': 'refrigerator', 'isthing': 1, 'color': [255, 112, 0]},
    101: {'name': 'washing_machine', 'isthing': 1, 'color': [143, 255, 0]},
    102: {'name': 'Microwave_oven', 'isthing': 1, 'color': [82, 0, 255]},
    103: {'name': 'fan', 'isthing': 1, 'color': [163, 255, 0]},
    104: {'name': 'curtain', 'isthing': 0, 'color': [255, 235, 0]},
    105: {'name': 'textiles', 'isthing': 0, 'color': [8, 184, 170]},
    106: {'name': 'clothes', 'isthing': 0, 'color': [133, 0, 255]},
    107: {'name': 'painting_or_poster', 'isthing': 1, 'color': [0, 255, 92]},
    108: {'name': 'mirror', 'isthing': 1, 'color': [184, 0, 255]},
    109: {'name': 'flower_pot_or_vase', 'isthing': 1, 'color': [255, 0, 31]},
    110: {'name': 'clock', 'isthing': 1, 'color': [0, 184, 255]},
    111: {'name': 'book', 'isthing': 0, 'color': [0, 214, 255]},
    112: {'name': 'tool', 'isthing': 0, 'color': [255, 0, 112]},
    113: {'name': 'blackboard', 'isthing': 0, 'color': [92, 255, 0]},
    114: {'name': 'tissue', 'isthing': 0, 'color': [0, 224, 255]},
    115: {'name': 'screen_or_television', 'isthing': 1, 'color': [112, 224, 255]},
    116: {'name': 'computer', 'isthing': 1, 'color': [70, 184, 160]},
    117: {'name': 'printer', 'isthing': 1, 'color': [163, 0, 255]},
    118: {'name': 'Mobile_phone', 'isthing': 1, 'color': [153, 0, 255]},
    119: {'name': 'keyboard', 'isthing': 1, 'color': [71, 255, 0]},
    120: {'name': 'other_electronic_product', 'isthing': 0, 'color': [255, 0, 163]},
    121: {'name': 'fruit', 'isthing': 0, 'color': [255, 204, 0]},
    122: {'name': 'food', 'isthing': 0, 'color': [255, 0, 143]},
    123: {'name': 'instrument', 'isthing': 1, 'color': [0, 255, 235]},
    124: {'name': 'train', 'isthing': 1, 'color': [133, 255, 0]}
}

class VideoVIPSegDataset(UnidVideoDataset):
    def __init__(self, T, max_frame_skip=3, resize=(480, 640), crop_scale=(0.6, 1.0), crop_ratio=(1.0,1.0), near_plane=1e-5, far_plane=65.0, full=True, fill_missing_tasks=True, load_tasks=None, **kwargs):
        super().__init__(T=T, max_frame_skip=max_frame_skip, resize=resize, crop_scale=crop_scale, crop_ratio=crop_ratio, near_plane=near_plane, far_plane=far_plane, **kwargs)
        self.config = DATASET_CONFIG["vipseg"]
        self.dataset_name = "vipseg"
        self.gt_tasks = {'rgb', 'semseg-vipseg', 'boundary'}
        self.load_tasks = load_tasks if load_tasks is None else set(load_tasks)    # if None, load all available tasks
        self.sem_categories = self._get_sem_categories()
        self.sem_color_code = self._construct_color_code()

        if full:
            self.df = pd.read_parquet(self._file_io(self.config["parquet_path"]))
            self.video_frames = self._organize_by_video()
            self.frame_list = self._create_frame_list()
            logging.info(f"[VIPSeg] found {len(self.frame_list)} frames across {len(self.video_frames)} videos.")
    
    def _get_sem_categories(self):
        """Organize frames by video name and sort by frame number."""
        categories = {
            k: {'name': v['name'], 'supercategory': '_', 'isthing': v['isthing']} for k, v in VIPSEG_CATEGORIES.items()
        }
        return categories

    def _construct_color_code(self):
        color_code = torch.zeros((max(VIPSEG_CATEGORIES.keys())+1,3), dtype=torch.float32)
        for k, v in VIPSEG_CATEGORIES.items():
            color_code[k] = torch.tensor(VIPSEG_CATEGORIES[k]['color'], dtype=torch.float32) / 255.0
        return color_code  # num_classes, 3
    
    def _is_gt_task(self, task_name):
        return task_name in self.gt_tasks
    
    def _is_load_task(self, task_name):
        assert self._is_gt_task(task_name), f"{task_name} not in gt tasks"
        return self.load_tasks is None or task_name in self.load_tasks

    def _load_panoptic_vipseg(self, data):
        if not self._is_load_task('semseg-vipseg') and not self._is_load_task('boundary'):
            return None
        pan_mask = np.array(Image.open(self._file_io(data['panoptic_path'])))
        instance_ids = np.unique(pan_mask)
        pan_info = []
        for inst_id in instance_ids:
            if inst_id == 0:
                continue  # skip background
            semantic_id = inst_id if inst_id <= 124 else inst_id // 100
            isthing = self.sem_categories[semantic_id]['isthing']
            pan_info.append({
                'id': int(inst_id),
                'category_id': int(semantic_id),
                'isthing': isthing,
            })

        return pan_mask, pan_info

def generate_parquet_file(split="train", input_root=None, output_root=None):
    DATA_DIR = str(dataset_root("vipseg", input_root))

    with open(osp.join(DATA_DIR, split+'.txt'), 'r') as f:
        video_list = sorted(x.strip() for x in f if x.strip())

    data_list = []
    for video in tqdm.tqdm(video_list):
        rgb_dir = osp.join(DATA_DIR, 'imgs', video)
        pan_dir = osp.join(DATA_DIR, 'panomasks', video)
        pseudo_depth_dir = osp.join(DATA_DIR, split+"_depth-pro_pred", video)
        pseudo_normal_dir = osp.join(DATA_DIR, split+"_dsine_pred", video)
        pseudo_kp_dir = osp.join(DATA_DIR, split+"_rtmo_pred", video)
        rgb_files = sorted(os.listdir(rgb_dir))
        for rgb_file in rgb_files:
            frame_number = int(rgb_file.split('_')[-1].split('.')[0])
            
            rgb_path = osp.join(rgb_dir, rgb_file)
            panoptic_path = osp.join(pan_dir, f'{frame_number:08d}.png')
            require_files(rgb_path, panoptic_path)

            pseudo_depth_path = osp.join(pseudo_depth_dir, f'{frame_number:08d}_depth.png')
            if not osp.exists(pseudo_depth_path):
                pseudo_depth_path = None
            
            pseudo_normal_path = osp.join(pseudo_normal_dir, f'{frame_number:08d}_normal.png')
            if not osp.exists(pseudo_normal_path):
                pseudo_normal_path = None
            
            pseudo_kp_path = osp.join(pseudo_kp_dir, f'{frame_number:08d}_keypoint.npz')
            if not osp.exists(pseudo_kp_path):
                pseudo_kp_path = None

            data_list.append({
                'rgb_path': rgb_path,
                'panoptic_path': panoptic_path,
                'pseudo_depth_path': pseudo_depth_path,
                'pseudo_normal_path': pseudo_normal_path,
                'pseudo_coco_panoptic_path': None, # never pseudo label since we have GT
                'pseudo_coco_keypoint_path': pseudo_kp_path,
                'video_name': video,
                'frame_id': frame_number,
            })

    return write_index(data_list, "vipseg", split, output_root)


if __name__ == "__main__":
    # generate_parquet_file()

    # sample and visualize:
    print("\nTesting image dataset...")
    image_dataset = VideoVIPSegDataset(T=4)
    print(f"Video dataset length: {len(image_dataset)}")
    # breakpoint()
    image_sample = image_dataset[101]
    # image_sample = image_dataset[1937]
    # torch.manual_seed(1); rand_proj = torch.rand(3,17)
    # instances = (torch.rand(10) * len(image_dataset)).long().tolist()
    # for ii in tqdm.tqdm(instances):
    #     x = []
    #     for i in range(5):
    #         image_sample = image_dataset[ii]
    #         rgb = ((image_sample['rgb'] + 1)/2).permute(1,2,0).numpy()
    #         depth = image_sample['depth'].permute(1,2,0).numpy()
    #         depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    #         depth = np.concatenate([depth]*3, axis=2)
    #         normal = ((image_sample['normal'] + 1)/2).permute(1,2,0).numpy()
    #         semseg = image_sample['semseg-coco']
    #         semantic_ids = torch.unique(semseg)
    #         semseg_vis = torch.zeros(rgb.shape)
    #         for sid in semantic_ids:
    #             semseg_vis[semseg==sid] = image_dataset.sem_color_code[sid.long()]
    #         semseg = semseg_vis.numpy()

    #         boundary = image_sample['boundary'].permute(1,2,0).numpy()
    #         boundary = np.concatenate([boundary]*3, axis=2)

    #         keypoint = rand_proj @ image_sample['keypoint-coco'].reshape(17, -1)  # 17, H, W
    #         keypoint = keypoint.reshape(3, *rgb.shape[:2]).permute(1,2,0).numpy()
    #         keypoint = (keypoint - keypoint.min()) / (keypoint.max() - keypoint.min() + 1e-8)

    #         out_img = np.hstack((rgb, depth, normal, semseg, boundary, keypoint)) * 255
    #         x.append(out_img)
    #     os.makedirs("data_vis", exist_ok=True)
    #     cv2.imwrite(f"data_vis/vipseg_sample_{ii}.png", np.vstack(x).astype(np.uint8))
