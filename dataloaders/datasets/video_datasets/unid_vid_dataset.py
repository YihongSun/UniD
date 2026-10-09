import os, cv2, io, json, random, tqdm
import os.path as osp
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataloaders.config import DATASET_CONFIG
from dataloaders.datasets.utils import generate_gaussian_heatmaps, panoptic2semseg, panoptic2boundary, sample_frame_indices, get_pan_data
from dataloaders.transforms.spatial_transform import SpatialTransform
from dataloaders.datasets.image_datasets.unid_img_dataset import UnidImageDataset


class UnidVideoDataset(UnidImageDataset):
    def __init__(self, T, max_frame_skip=3, resize=(480, 640), crop_scale=(0.8, 1.0), crop_ratio=(1.0,1.0), near_plane=1e-5, far_plane=65.0, fill_missing_tasks=True, **kwargs):
        super().__init__(resize=resize, crop_scale=crop_scale, crop_ratio=crop_ratio, near_plane=near_plane, far_plane=far_plane, fill_missing_tasks=fill_missing_tasks, **kwargs)
        self.T = T                              # number of frames in a sequence
        self.max_frame_skip = max_frame_skip    # max frame skip between consecutive frames

    def _organize_by_video(self):
        """Organize frames by video name and sort by frame number."""
        video_frames = defaultdict(list)
        
        # self.df needs to be initialized by child class
        for _, row in self.df.iterrows():
            video_frames[row['video_name']].append(dict(row))
        
        # sort by frame number
        [v.sort(key=lambda x: x['frame_id']) for k,v in video_frames.items()]
        
        return dict(video_frames)
    
    def _create_frame_list(self):
        """Create a flat list of all frames with video context."""
        frame_list = []
        for video_name, frames in self.video_frames.items():
            if len(frames) < self.T:
                continue  # skip short videos
            frame_list += [(video_name, ii) for ii, frame_data in enumerate(frames)]
        return frame_list

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx=None):
        if idx is None:
            idx = np.random.randint(len(self.df))

        video_name, frame_idx = self.frame_list[idx]
        video_data = self.video_frames[video_name]
        video_length = len(video_data)
        sampled_indices = sample_frame_indices(frame_idx, video_length, self.T, self.max_frame_skip)

        loaded = []
        for si in sampled_indices:
            data = video_data[si]
            loaded.append(self.load_data(data))
            
        # list of dict -> dict of list
        loaded = {
            k: [loaded[t][k] for t in range(self.T)] for k in loaded[0].keys()
        }
        transformed = self.transform(loaded)
        transformed = self.blur_transform(transformed)

        output_list, data_sup_list = [], []
        for t in range(self.T):
            output, data_sup = self.prepare_output(
                {k: transformed[k][t] for k in transformed.keys()}
            )
            output_list.append(output)
            data_sup_list.append(data_sup)
        
        # list of dict -> dict of stacked tensor
        output_stacked = {k: torch.stack([output_list[t][k] for t in range(self.T)]) for k in output_list[0].keys()}
        data_sup_stacked = {k: torch.tensor([data_sup_list[t][k] for t in range(self.T)], dtype=torch.uint8) for k in data_sup_list[0].keys()}

        output_stacked.update({
            'supervision': data_sup_stacked,
            'dataset': self.dataset_name,
            'instance': idx,
        })

        return output_stacked