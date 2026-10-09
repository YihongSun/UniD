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

SPRING_BASELINE = 0.065

def get_depth(disp1, focal_length, baseline=SPRING_BASELINE):
    """
    get depth from reference frame disparity and camera intrinsics
    """
    valid_mask = disp1 > 0
    depth = np.zeros_like(disp1, dtype=np.float32)
    
    # Only compute depth for valid disparity values
    if np.any(valid_mask):
        # Add a small epsilon to prevent division by zero, but only for valid disparities
        depth[valid_mask] = focal_length * baseline / (disp1[valid_mask] + 1e-6)
        # Clip to reasonable depth range
        depth = np.clip(depth, 0, 1000.0)
    
    return depth

class VideoSpringDataset(UnidVideoDataset):
    def __init__(self, T, max_frame_skip=3, resize=(480, 640), crop_scale=(0.6, 1.0), crop_ratio=(1.0,1.0), near_plane=1e-5, far_plane=65.0, fill_missing_tasks=True, load_tasks=None, **kwargs):
        super().__init__(T=T, max_frame_skip=max_frame_skip, resize=resize, crop_scale=crop_scale, crop_ratio=crop_ratio, near_plane=near_plane, far_plane=far_plane, **kwargs)
        self.config = DATASET_CONFIG["spring"]
        self.df = pd.read_parquet(self._file_io(self.config["parquet_path"]))
        self.video_frames = self._organize_by_video()
        self.frame_list = self._create_frame_list()

        self.dataset_name = "spring"
        self.gt_tasks = {'rgb', 'depth'}
        self.load_tasks = load_tasks if load_tasks is None else set(load_tasks)    # if None, load all available tasks
        logging.info(f"[Spring] found {len(self.frame_list)} frames across {len(self.video_frames)} videos.")
    
    def _is_gt_task(self, task_name):
        return task_name in self.gt_tasks
    
    def _is_load_task(self, task_name):
        assert self._is_gt_task(task_name), f"{task_name} not in gt tasks"
        return self.load_tasks is None or task_name in self.load_tasks

    def _load_depth(self, data):
        if not self._is_load_task('depth'):
            return None
        disp_data = h5py.File(self._file_io(data['depth_path']), 'r')['disparity'][()]
        intrinsics = np.array(data['intrinsics']).reshape(3, 3)
        depth_data = get_depth(disp_data, intrinsics[0, 0]).astype(np.float32)
        depth_image = Image.fromarray(depth_data)
        return depth_image

def generate_parquet_file(split="train", input_root=None, output_root=None):
    DATA_DIR = str(dataset_root("spring", input_root))

    data_list = []
    scenes = sorted(os.listdir(osp.join(DATA_DIR, split)))
    for scene in tqdm.tqdm(scenes):
        rgb_dir = osp.join(DATA_DIR, split, scene, 'frame_left')
        depth_dir = osp.join(DATA_DIR, split, scene, 'disp1_left')
        intrinsics_file = osp.join(DATA_DIR, split, scene, 'cam_data/intrinsics.txt')
        intrinsics = np.loadtxt(intrinsics_file)
        pseudo_pan_dir = osp.join(DATA_DIR, split+"_oneformer-pred", scene, 'frame_left')
        pseudo_kp_dir = osp.join(DATA_DIR, split+"_rtmo-pred", scene, 'frame_left')
        rgb_files = sorted(os.listdir(rgb_dir))
        for rgb_file in rgb_files:
            frame_number = int(rgb_file.split('_')[-1].split('.')[0]) - 1
            depth_file = rgb_file.replace('.png', '.dsp5').replace('frame', 'disp1')
            rgb_path = osp.join(rgb_dir, rgb_file)
            depth_path = osp.join(depth_dir, depth_file)
            require_files(rgb_path, depth_path)

            pseudo_pan_path = osp.join(pseudo_pan_dir, rgb_file.removesuffix('.png')+'_panoptic.png')
            if not osp.exists(pseudo_pan_path):
                pseudo_pan_path = None
            
            pseudo_kp_path = osp.join(pseudo_kp_dir, rgb_file.removesuffix('.png')+'_keypoint.npz')
            if not osp.exists(pseudo_kp_path):
                pseudo_kp_path = None

            intrinsic_matrix = np.eye(3, dtype=np.float32)
            intrinsic_matrix[0, 0] = intrinsics[frame_number][0]
            intrinsic_matrix[1, 1] = intrinsics[frame_number][1]
            intrinsic_matrix[0, 2] = intrinsics[frame_number][2]
            intrinsic_matrix[1, 2] = intrinsics[frame_number][3]    
            intrinsic_matrix_flat = intrinsic_matrix.flatten().tolist()

            data_list.append({
                'rgb_path': rgb_path,
                'depth_path': depth_path,
                # 'normal_path': normal_path,   # todo: implement in the future
                'pseudo_normal_path': None,     # ignroe for now
                'pseudo_coco_panoptic_path': pseudo_pan_path,
                'pseudo_coco_keypoint_path': pseudo_kp_path,
                'video_name': scene,
                'frame_id': frame_number,
                'intrinsics': intrinsic_matrix_flat,
            })

    return write_index(data_list, "spring", split, output_root)


if __name__ == "__main__":
    # generate_parquet_file()

    # sample and visualize:
    print("\nTesting image dataset...")
    image_dataset = VideoSpringDataset(T=5)
    print(f"Video dataset length: {len(image_dataset)}")
    # breakpoint()
    # image_sample = image_dataset[101]
    # image_sample = image_dataset[1937]
    torch.manual_seed(1); rand_proj = torch.rand(3,17)
    instances = (torch.rand(10) * len(image_dataset)).long().tolist()
    instances = [3856]
    for ii in tqdm.tqdm(instances):
        image_sample = image_dataset[ii]
        x = []
        for i in range(5):
            rgb = ((image_sample['rgb'][i] + 1)/2).permute(1,2,0).numpy()
            depth = image_sample['depth'][i].permute(1,2,0).numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = np.concatenate([depth]*3, axis=2)
            normal = ((image_sample['normal'][i] + 1)/2).permute(1,2,0).numpy()
            semseg = image_sample['semseg-coco'][i]
            semantic_ids = torch.unique(semseg)
            semseg_vis = torch.zeros(rgb.shape)
            for sid in semantic_ids:
                semseg_vis[semseg==sid] = image_dataset.sem_color_code[sid.long()]
            semseg = semseg_vis.numpy()

            boundary = image_sample['boundary'][i].permute(1,2,0).numpy()
            boundary = np.concatenate([boundary]*3, axis=2)

            keypoint = rand_proj @ image_sample['keypoint-coco'][i].reshape(17, -1)  # 17, H, W
            keypoint = keypoint.reshape(3, *rgb.shape[:2]).permute(1,2,0).numpy()
            keypoint = (keypoint - keypoint.min()) / (keypoint.max() - keypoint.min() + 1e-8)

            out_img = np.hstack((rgb, depth, normal, semseg, boundary, keypoint)) * 255
            x.append(out_img)
        os.makedirs("data_vis", exist_ok=True)
        cv2.imwrite(f"data_vis/spring_sample_{ii}.png", np.vstack(x).astype(np.uint8))
