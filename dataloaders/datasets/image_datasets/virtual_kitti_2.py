import os, cv2, io, json, random, logging, tqdm
import os.path as osp
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataloaders.config import DATASET_CONFIG
from dataloaders.preparation import dataset_root, require_files, write_index
from dataloaders.datasets.utils import generate_gaussian_heatmaps
from dataloaders.transforms.spatial_transform import SpatialTransform
from dataloaders.datasets.image_datasets.unid_img_dataset import UnidImageDataset

# Dataset specific parameters
OG_VKITTI_FOCAL_LENGTH = 725.0087

class ImageVKITTI2Dataset(UnidImageDataset):
    def __init__(self, resize=(480, 640), crop_scale=(0.6, 1.0), crop_ratio=(1.0,1.0), near_plane=1e-5, far_plane=65.0, fill_missing_tasks=True, load_tasks=None, **kwargs):
        super().__init__(resize=resize, crop_scale=crop_scale, crop_ratio=crop_ratio, near_plane=near_plane, far_plane=far_plane, fill_missing_tasks=fill_missing_tasks, **kwargs)
        self.config = DATASET_CONFIG["virtual_kitti_2"]
        self.df = pd.read_parquet(self._file_io(self.config["parquet_path"]))
        self.dataset_name = "virtual_kitti_2"
        self.gt_tasks = {'rgb', 'depth', 'normal'}
        self.load_tasks = load_tasks if load_tasks is None else set(load_tasks)    # if None, load all available tasks
        logging.info(f"[VKITTI2] found {len(self.df)} images.")
    
    def _is_gt_task(self, task_name):
        return task_name in self.gt_tasks
    
    def _is_load_task(self, task_name):
        assert self._is_gt_task(task_name), f"{task_name} not in gt tasks"
        return self.load_tasks is None or task_name in self.load_tasks

    def _load_depth(self, data):
        if not self._is_load_task('depth'):
            return None
        depth_image = Image.open(self._file_io(data['depth_path']))
        depth_image = np.array(depth_image) / 100 # cm to meters
        depth_image = Image.fromarray(depth_image)
        return depth_image

    def _load_normal(self, data, depth_image, align_cam_normal=True):
        if not self._is_load_task('normal'):
            return None
        normal_image = Image.open(self._file_io(data['normal_path'])).convert('RGB')
        return normal_image

def generate_parquet_file(split="train", input_root=None, output_root=None):
    if split != "train":
        raise ValueError("Virtual KITTI 2 builder indexes the configured training scenes only; use --split train.")
    SCRIPT_DIR = str(dataset_root("virtual_kitti_2", input_root))

    scenes = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
    weather_conditions = ["morning", "fog", "rain", "sunset", "overcast"]
    cameras = ["Camera_0", "Camera_1"]
    vkitti2_rgb_path = osp.join(SCRIPT_DIR, "vkitti_2.0.3_rgb")
    vkitti2_depth_path =  osp.join(SCRIPT_DIR, "vkitti_2.0.3_depth")
    vkitti2_normal_path = osp.join(SCRIPT_DIR, "vkitti_DAG_normals")
    vkitti2_pseudo_pan_path = osp.join(SCRIPT_DIR, "vkitti_2.0.3_oneformer-pred")
    vkitti2_pseudo_kp_path =  osp.join(SCRIPT_DIR, "vkitti_2.0.3_rtmo-pred")


    data_list = []
    for scene in scenes:
        for weather in weather_conditions:
            for camera in cameras:
                image_files = sorted(os.listdir(osp.join(vkitti2_rgb_path, scene, weather, "frames", "rgb", camera)))
                for image_file in image_files:
                    frame_id = osp.splitext(image_file)[0].removeprefix("rgb_")
                    rgb_path = osp.join(vkitti2_rgb_path, scene, weather, "frames", "rgb" ,camera, f'rgb_{frame_id}.jpg')
                    depth_path = osp.join(vkitti2_depth_path, scene, weather, "frames","depth" , camera, f'depth_{frame_id}.png')
                    normal_path = osp.join(vkitti2_normal_path, scene, weather, "frames", "normal", camera, f'normal_{frame_id}.png')
                    require_files(rgb_path, depth_path, normal_path)

                    local_pan_path = osp.join(vkitti2_pseudo_pan_path, scene, weather, "frames", "rgb", camera, f'rgb_{frame_id}_panoptic.png')
                    if osp.exists(local_pan_path):
                        panoptic_path = local_pan_path
                    else:
                        panoptic_path = None
                    
                    local_kp_path = osp.join(vkitti2_pseudo_kp_path, scene, weather, "frames", "rgb", camera, f'rgb_{frame_id}_keypoint.npz')
                    if osp.exists(local_kp_path):
                        keypoint_path = local_kp_path
                    else:
                        keypoint_path = None

                    data_list.append({
                        'rgb_path': rgb_path,
                        'depth_path': depth_path,
                        'normal_path': normal_path,
                        'pseudo_coco_panoptic_path': panoptic_path,
                        'pseudo_coco_keypoint_path': keypoint_path,
                    })
    return write_index(data_list, "virtual_kitti_2", split, output_root)


if __name__ == "__main__":
    # generate_parquet_file()

    # sample and visualize:
    print("\nTesting image dataset...")
    image_dataset = ImageVKITTI2Dataset()
    print(f"Image dataset length: {len(image_dataset)}")
    breakpoint()
    # image_sample = image_dataset[101]
    # image_sample = image_dataset[1937]
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
        cv2.imwrite(f"data_vis/virtual_kitti_2_sample_{ii}.png", np.vstack(x).astype(np.uint8))
