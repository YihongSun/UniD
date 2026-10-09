import random, cv2
import numpy as np
from PIL import Image
import torch

def generate_gaussian_heatmaps(keypoints_list, image_shape, sigma=1.0, kernel_size=None, num_keypoints=17):
    """ Generate Gaussian heatmaps for keypoints.
    """
    H, W = image_shape
    if kernel_size is None:
        kernel_size = int(6 * sigma)  # 3-sigma rule on each side
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    radius = kernel_size // 2
    
    # Initialize heatmaps
    heatmaps = np.zeros((num_keypoints, H, W), dtype=np.float32)
    
    # Pre-compute Gaussian kernel
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, y)
    gaussian_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Process each person's keypoints
    for keypoints in keypoints_list:
        kpts = np.array(keypoints).reshape(17, 3)  # Shape: (17, 3)
        
        for joint_idx in range(num_keypoints):
            x, y, v = kpts[joint_idx]
            
            # Skip if not labeled (v == 0)
            if v == 0:
                continue
            
            # Round to nearest pixel
            x_int = int(round(x))
            y_int = int(round(y))
            
            # Calculate bounds for the heatmap
            x_start = max(0, x_int - radius)
            x_end = min(W, x_int + radius + 1)
            y_start = max(0, y_int - radius)
            y_end = min(H, y_int + radius + 1)
            
            # Calculate corresponding kernel bounds
            kernel_x_start = radius - (x_int - x_start)
            kernel_x_end = radius + (x_end - x_int)
            kernel_y_start = radius - (y_int - y_start)
            kernel_y_end = radius + (y_end - y_int)  # Fixed: was y_int - y_int
            
            # Place Gaussian (take maximum for overlapping regions)
            heatmaps[joint_idx, y_start:y_end, x_start:x_end] = np.maximum(
                heatmaps[joint_idx, y_start:y_end, x_start:x_end],
                gaussian_kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]
            )
    
    return torch.from_numpy(heatmaps)


def panoptic2semseg(panoptic_data):
    ''' Convert panoptic segmentation to semantic segmentation.
    '''
    if panoptic_data is None:
        return None
    panoptic_mask, panoptic_info = panoptic_data
    semantic_seg = np.zeros_like(panoptic_mask, dtype=np.uint8)
    for ann in panoptic_info:
        semantic_id = ann['category_id']
        mask = panoptic_mask == ann['id']
        semantic_seg[mask] = semantic_id
    semantic_seg = Image.fromarray(semantic_seg.astype(np.uint8)).convert("P")
    return semantic_seg


def panoptic2boundary(panoptic_data):
    ''' Convert panoptic segmentation to boundary map.
    '''
    if panoptic_data is None:
        return None
    panoptic_mask, panoptic_info = panoptic_data
    boundary_contours = []
    instance_seg = np.zeros_like(panoptic_mask, dtype=np.uint8)
    for ann in panoptic_info:
        if ann['isthing']:
            mask = panoptic_mask == ann['id']
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundary_contours += [contour.flatten().tolist() for contour in contours]
    return boundary_contours

def get_pan_data(pan_datas):
    presence = [x is not None for x in pan_datas]
    num_present = np.sum(presence)
    if num_present == 0:
        return None
    elif num_present == 1:
        return pan_datas[presence.index(True)]
    else:
        raise ValueError("Multiple panoptic data sources present.")

def sample_frame_indices(start_idx, num_frames, seq_length, max_frame_skip, prob_reverse_time=0.5):
    """
    Sample frame indices
    """

    sampled_frames = [start_idx]
    acceptable_set = set(
        range(
            max(0, sampled_frames[-1] - max_frame_skip),
            min(num_frames, sampled_frames[-1] + max_frame_skip + 1),
        )
    ).difference(set(sampled_frames))
    while len(sampled_frames) < seq_length:
        idx = np.random.choice(list(acceptable_set))
        sampled_frames.append(idx)
        new_set = set(
            range(
                max(0, sampled_frames[-1] - max_frame_skip),
                min(num_frames, sampled_frames[-1] + max_frame_skip + 1),
            )
        )
        acceptable_set = acceptable_set.union(new_set).difference(set(sampled_frames))

    sampled_frames = sorted(sampled_frames)
    if np.random.rand() < prob_reverse_time:
        # Reverse time
        sampled_frames = sampled_frames[::-1]

    return sampled_frames