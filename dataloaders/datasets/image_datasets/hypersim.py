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
OG_HYPER_FOCAL_LENGTH = 886.81

# Helper functions (will move to utils.py if needed elsewhere)
def creat_uv_mesh(H, W):
    y, x = np.meshgrid(np.arange(0, H, dtype=np.float64), np.arange(0, W, dtype=np.float64), indexing='ij')
    meshgrid = np.stack((x,y))
    ones = np.ones((1,H*W), dtype=np.float64)
    xy = meshgrid.reshape(2, -1)
    return np.concatenate([xy, ones], axis=0)

def align_normals(normal, depth, K, H, W):
    '''
    Orientation of surface normals in hypersim is not always consistent
    see https://github.com/apple/ml-hypersim/issues/26
    '''
    # Some Hypersim normals are not properly oriented towards the camera.
    # The align_normals and creat_uv_mesh functions are from GeoWizard
    # https://github.com/fuxiao0719/GeoWizard/blob/5ff496579c6be35d9d86fe4d0760a6b5e6ba25c5/geowizard/training/dataloader/file_io.py#L115
    # inv K
    K = np.array([[K[0],    0, K[2]],
                    [   0, K[1], K[3]],
                    [   0,    0,    1]])
    inv_K = np.linalg.inv(K)
    # reprojection depth to camera points
    xy = creat_uv_mesh(H, W)
    points = np.matmul(inv_K[:3, :3], xy).reshape(3, H, W)
    points = depth * points
    points = points.transpose((1,2,0))
    # align normal
    orient_mask = np.sum(normal * points, axis=2) > 0
    normal[orient_mask] *= -1
    return normal


class ImageHypersimDataset(UnidImageDataset):
    def __init__(self, resize=(480, 640), crop_scale=(0.6, 1.0), crop_ratio=(1.0,1.0), near_plane=1e-5, far_plane=65.0, fill_missing_tasks=True, load_tasks=None, **kwargs):
        super().__init__(resize=resize, crop_scale=crop_scale, crop_ratio=crop_ratio, near_plane=near_plane, far_plane=far_plane, fill_missing_tasks=fill_missing_tasks, **kwargs)
        self.config = DATASET_CONFIG["hypersim"]
        self.df = pd.read_parquet(self._file_io(self.config["parquet_path"]))
        self.dataset_name = "hypersim"
        self.gt_tasks = {'rgb', 'depth', 'normal', 'albedo', 'illum'}
        self.load_tasks = load_tasks if load_tasks is None else set(load_tasks)    # if None, load all available tasks
        logging.info(f"[Hypersim] found {len(self.df)} images.")
    
    def _is_gt_task(self, task_name):
        return task_name in self.gt_tasks
    
    def _is_load_task(self, task_name):
        assert self._is_gt_task(task_name), f"{task_name} not in gt tasks"
        return self.load_tasks is None or task_name in self.load_tasks

    def _load_depth(self, data):
        if not self._is_load_task('depth'):
            return None
        depth_image = Image.open(self._file_io(data['depth_path']))
        depth_image = np.array(depth_image) / 1000 # mm to meters
        depth_image = Image.fromarray(depth_image)
        return depth_image

    def _load_normal(self, data, depth_image, align_cam_normal=True):
        if not self._is_load_task('normal'):
            return None
        normal_image = Image.open(self._file_io(data['normal_path'])).convert('RGB')
        if align_cam_normal:
            # align normals towards camera
            normal_array = (np.array(normal_image) / 255.0) * 2.0 - 1.0
            H, W = normal_array.shape[:2]
            normal_array[:,:,1:] *= -1
            normal_array = align_normals(normal_array, np.array(depth_image), [OG_HYPER_FOCAL_LENGTH, OG_HYPER_FOCAL_LENGTH,W/2, H/2], H, W) * -1
            normal_image = Image.fromarray(((normal_array + 1.0) / 2.0 * 255).astype(np.uint8))

        return normal_image
    
    def _load_albedo(self, data):
        if not self._is_load_task('albedo'):
            return None
        albedo_image = Image.open(self._file_io(data['albedo_path'])).convert('RGB')
        return albedo_image

    def _load_illum(self, data):
        if not self._is_load_task('illum'):
            return None
        illum_image = Image.open(self._file_io(data['illum_path'])).convert('RGB')
        return illum_image

def generate_parquet_file(split="train", input_root=None, output_root=None):
    SCRIPT_DIR = str(dataset_root("hypersim", input_root))
    pseudo_root = osp.dirname(SCRIPT_DIR)
    df = pd.read_csv(osp.join(SCRIPT_DIR, split, "filename_meta_train.csv"))
    df = df.sort_values("rgb_path")

    data_list = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing annotations"):
        if row['included_in_public_release'] and (row['split_partition_name'] == split):
            rgb_path = osp.join(SCRIPT_DIR, split, row['rgb_path'])
            depth_path = osp.join(SCRIPT_DIR, split, row['depth_path'])
            normal_dir = osp.join(pseudo_root, 'normals', row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_preview', f'frame.{str(row["frame_id"]).zfill(4)}.normal_cam.png')
            fname = osp.splitext(row['rgb_path'])[0]
            albedo_path = osp.join(SCRIPT_DIR, split, fname.replace('rgb', 'albedo') + '.png')
            illum_path = osp.join(SCRIPT_DIR, split, fname.replace('rgb', 'illum') + '.png')
            require_files(rgb_path, depth_path, normal_dir, albedo_path, illum_path)

            local_pan_path = osp.join(pseudo_root, "oneformer-pred", fname+'_panoptic.png')
            if osp.exists(local_pan_path):
                panoptic_path = local_pan_path
            else:
                panoptic_path = None

            local_kp_path = osp.join(pseudo_root, "rtmo-pred", fname+'_keypoint.npz')
            if osp.exists(local_kp_path):
                keypoint_path = local_kp_path
            else:
                keypoint_path = None
            
            data_list.append({
                'rgb_path': rgb_path,
                'depth_path': depth_path,
                'normal_path': normal_dir,
                'albedo_path': albedo_path,
                'illum_path': illum_path,
                'pseudo_coco_panoptic_path': panoptic_path,
                'pseudo_coco_keypoint_path': keypoint_path,
            })

    return write_index(data_list, "hypersim", split, output_root)


if __name__ == "__main__":
    # generate_parquet_file()

    # sample and visualize:
    print("\nTesting image dataset...")
    image_dataset = ImageHypersimDataset(fill_missing_tasks=False, load_tasks=['albedo', 'illum'])
    print(f"Image dataset length: {len(image_dataset)}")
    torch.manual_seed(1); rand_proj = torch.rand(3,17)
    instances = (torch.rand(10) * len(image_dataset)).long().tolist()
    for ii in tqdm.tqdm(instances):
        x = []
        for i in range(5):
            image_sample = image_dataset[ii]
            rgb = ((image_sample['rgb'] + 1)/2).permute(1,2,0).numpy()
            albedo = ((image_sample['albedo'] + 1)/2).permute(1,2,0).numpy()
            illum = ((image_sample['illum'] + 1)/2).permute(1,2,0).numpy()
            out_img = np.hstack((rgb, albedo, illum)) * 255
            x.append(out_img)
        os.makedirs("data_vis", exist_ok=True)
        cv2.imwrite(f"data_vis/hypersim_albedo_illum_sample_{ii}.png", np.vstack(x).astype(np.uint8))
        


    # # sample and visualize:
    # print("\nTesting image dataset...")
    # image_dataset = ImageHypersimDataset()
    # print(f"Image dataset length: {len(image_dataset)}")
    # breakpoint()
    # # image_sample = image_dataset[101]
    # # image_sample = image_dataset[1937]
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
    #     cv2.imwrite(f"data_vis/hypersim_sample_{ii}.png", np.vstack(x).astype(np.uint8))
