import logging
import os
import sys
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F


def get_training_vis(batch, current_estimate_proj, ii, proj_names, proj_color_codes):
    """ Get visualization for the current training estimate of all projections concatenated together, along with the RGB input.
    This is used for visualizing the current estimate of all projections during training, concatenated together for easier comparison, along with the RGB input.
    """
    dev_bs, T, _, _, _ = batch["rgb"].shape  # B, T, C, H, W

    gt_v = [torch.cat([_ for _ in batch["rgb"][ii]], dim=2)] # 3, H, W*T
    pd_v = [torch.cat([_ for _ in batch["rgb"][ii]], dim=2)] # 3, H, W*T

    for proj_name in proj_names:
        gt = batch[proj_name][ii]
        pd = current_estimate_proj[proj_name][ii].detach()

        gt_v.append(get_proj_only_vis(proj_name, gt, proj_color_codes))
        pd_v.append(get_proj_only_vis(proj_name, pd, proj_color_codes))
    
    vis = torch.cat([torch.cat(pd_v, dim=1),torch.cat(gt_v, dim=1)], dim=2) # 3, H*(1+P), W*T*2
    vis_np = (vis.permute(1,2,0).cpu().numpy() + 1) / 2.0 * 255
    vis_np = vis_np.clip(0, 255).astype(np.uint8)                           # H*(1+P), W*T*2, 3
    return vis_np


def get_latent_vis(current_estimate, batch, ii):
    (dev_bs, T, _, H, W) = batch["rgb"].shape  # B, T, C, H, W
    v = [
        torch.cat([_ for _ in batch["rgb"][ii]], dim=2),  # inp_img C, H, W*T, 
        torch.cat([_ for _ in proj_pca(current_estimate[ii])[0]], dim=2),  # pred_img, C, h, w*T
    ]
    v[1] = F.interpolate(v[1][None], size=(H, W*T), mode='nearest')[0].detach().cpu()

    vis_np = (torch.cat(v, dim=1).permute(1,2,0).cpu().numpy() + 1) / 2.0 * 255
    vis_np = vis_np.clip(0, 255).astype(np.uint8)
    return vis_np


def get_proj_only_vis(proj_name, current_estimate_proj, proj_color_codes):
    """ Get visualization for a single projection (e.g. depth, normal, semseg) without concatenating with other projections.
    This is used for visualizing the current estimate of each projection during training, without needing to concatenate with GT or other projections.
    """
    pred = current_estimate_proj                                # T, N, H, W
    if len(pred.shape) == 3:
        pred = pred[:, None]                                    # T, 1, H, W (hacky for gt)
    T, N, H, W = pred.shape
    if 'depth' in proj_name:
        pred_aligned = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)  # to [0, 1]
        pred_vis = colorize_depth_maps(
            pred_aligned.detach().cpu(), 0, 1
        ) * 2.0 - 1.0                                           # to [-1, 1]
    elif 'normal' in proj_name:
        pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True) + 1e-5
        pred_vis = torch.clip(pred, -1.0, 1.0)
    elif 'boundary' in proj_name:
        pred_vis = torch.clip(pred, 0.0, 1.0) * 2.0 - 1.0       # to [-1, 1]
        pred_vis = pred_vis.repeat(1,3,1,1)                     # to 3 channels
    elif 'albedo' in proj_name or 'illum' in proj_name or 'restore' in proj_name:
        pred_vis = torch.clip(pred, -1.0, 1.0)
    elif 'semseg' in proj_name or 'matseg' in proj_name or 'partseg' in proj_name:
        color_codes = proj_color_codes[proj_name]
        if N == color_codes.shape[0]:
            pred_mask = torch.argmax(pred, dim=1)               # T, H, W
        else:
            pred_mask = pred[:,0]                               # hacky to allow gt to pass in
        pred_vis = torch.zeros((T, H, W, 3))
        sem_ids = pred_mask.unique().tolist()
        for sem_id in sem_ids:
            pred_vis[pred_mask==sem_id] = color_codes[sem_id]
        pred_vis = pred_vis.permute(0,3,1,2) * 2.0 - 1.0        # to [-1, 1]
    elif 'keypoint' in proj_name:
        pred_resh = pred.permute(1,0,2,3).reshape(N, -1).cpu()      # N, T, H, W
        pred_vis = proj_color_codes[proj_name] @ pred_resh     # 3, T*H*W
        pred_vis = pred_vis.reshape(3, T, H, W).permute(1,0,2,3)    # T, 3, H, W
        pred_vis = torch.clip(pred_vis, 0.0, 1.0) * 2.0 - 1.0       # to [-1, 1]

    pred_vis = torch.cat([_ for _ in pred_vis], dim=2)      # 3, H, W*T
    return pred_vis.detach().cpu()


def proj_pca(x, target_dim=3):
    """ Project dim=1 of x (T, N, H, W) to target_dim using PCA.
    """
    T, N, H, W = x.shape
    if N == target_dim:
        return x
    x_inp = x.permute(0,2,3,1).reshape(-1, N)       # T*H*W, N
    mu = x_inp.mean(dim=-1, keepdim=True)
    x_centered = x_inp - mu
    U, S, V = torch.pca_lowrank(x_centered, q=target_dim, center=False, niter=2)
    x_reduced = torch.matmul(x_centered, V)         # T*H*W, target_dim

    x_out = x_reduced.reshape(T,H,W,3).permute(0,3,1,2)
    x_out = torch.clip(x_out, -1.0, 1.0)
    return x_out, (mu, V)


def proj_pca_precomp(x, mu, V):
    """ Project dim=1 of x (T, N, H, W) to target_dim using precomputed PCA parameters.
    """
    T, N, H, W = x.shape
    x_inp = x.permute(0,2,3,1).reshape(-1, N)       # T*H*W, N
    x_centered = x_inp - mu
    x_reduced = torch.matmul(x_centered, V)         # T*H*W, target_dim
    x_out = x_reduced.reshape(T,H,W,3).permute(0,3,1,2)
    x_out = torch.clip(x_out, -1.0, 1.0)
    return x_out


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def overwrite_args(args, overwrite_list):
    """ Overwrite config arguments based on command line input
    """
    if overwrite_list is None:
        return
    
    for item in overwrite_list:
        if '=' not in item:
            print(f"[OW:FAILURE] Invalid overwrite item '{item}'. Skipping.")
            continue
        key, value = item.split('=', 1)
        keys = key.split('.')
        d = args
        for k in keys[:-1]:
            if not hasattr(d, k):
                print(f"[OW:FAILURE] Key '{k}' not found in config. Skipping.")
                d = None
                break
            d = getattr(d, k)
        if d is not None and hasattr(d, keys[-1]):
            current_value = getattr(d, keys[-1])
            try:
                # Try to convert value to the same type as current_value
                if isinstance(current_value, bool):
                    value_converted = value.lower() in ('true', '1', 'yes')
                else:
                    value_converted = type(current_value)(value)
                setattr(d, keys[-1], value_converted)
                print(f"[OW:SUCCESS] Overwrote config '{key}' with {type(value_converted).__name__} '{value_converted}'.")
            except Exception as e:
                print(f"[OW:FAILURE] Failed to convert value '{value}' to type {type(current_value).__name__} for key '{key}'. Error: {e}. Skipping.")
        else:
            print(f"[OW:FAILURE] Key '{keys[-1]}' not found in config. Skipping.")