import random
import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode, RandomResizedCrop
import torchvision.transforms.functional as TF
from dataclasses import dataclass

@dataclass
class SpatialTransformParams:
    """Container for transformation parameters that can be applied to multiple samples."""

    flip: bool
    crop_i: int
    crop_j: int
    crop_h: int
    crop_w: int
    size: int
    org_size: tuple

class SpatialTransform:
    def __init__(
        self,
        size,
        prob_flip,
        crop_scale,
        crop_ratio,
    ):
        self.size = size
        self.prob_flip = prob_flip
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio

    def generate_params(self, img) -> SpatialTransformParams:
        """Generate transformation parameters for a given image."""
        # Random Horizontal Flip
        flip = random.random() < self.prob_flip

        # Random Resized Crop
        i, j, h, w = RandomResizedCrop.get_params(img, scale=self.crop_scale, ratio=self.crop_ratio)

        return SpatialTransformParams(
            flip=flip,
            crop_i=i,
            crop_j=j,
            crop_h=h,
            crop_w=w,
            size=self.size,
            org_size=img.size,
        )

    def apply_params_img(self, img, params: SpatialTransformParams, interpolation: InterpolationMode):
        if img is None:
            return None
        if params.flip:
            img = TF.hflip(img)
        w_factor, h_factor = img.size[0] / params.org_size[0], img.size[1] / params.org_size[1]
        img = TF.resized_crop(
            img,
            int(params.crop_i*h_factor),
            int(params.crop_j*w_factor),
            int(params.crop_h*h_factor),
            int(params.crop_w*w_factor),
            size=params.size,
            interpolation=interpolation,
        )
        return img

    def apply_params_boundary(self, boundary_data, params: SpatialTransformParams, dim: tuple):
        """Apply transformation parameters to boundary coordinates."""
        if boundary_data is None:
            return None
        H, W = dim
        new_H, new_W = params.size
        scale_x = new_W / params.crop_w
        scale_y = new_H / params.crop_h
        new_contours = []
        for contour in boundary_data:
            contour = np.array(contour).reshape(-1, 2)
            if params.flip:
                contour[:, 0] = W - contour[:, 0]
            contour[:, 0] = (contour[:, 0] - params.crop_j) * scale_x
            contour[:, 1] = (contour[:, 1] - params.crop_i) * scale_y
            # filter out points that are out of bounds
            contour = contour[
                (contour[:, 0] >= 0) & (contour[:, 0] < new_W) &
                (contour[:, 1] >= 0) & (contour[:, 1] < new_H)
            ]
            new_contours.append(contour.flatten().tolist())
        return new_contours

    def apply_params_keypoint(self, keypoint_data, params: SpatialTransformParams, dim: tuple):
        """Apply transformation parameters to keypoint coordinates."""
        if keypoint_data is None:
            return None
        H, W = dim
        new_H, new_W = params.size
        scale_x = new_W / params.crop_w
        scale_y = new_H / params.crop_h
        new_keypoints = []
        for person in keypoint_data:
            kpts = np.array(person).reshape(17, 3)  # Shape: (17, 3)
            if params.flip:
                kpts[:, 0] = W - kpts[:, 0]
                # swap left-right parts
                left_right_pairs = [
                    (1, 2), (3, 4), (5, 6), (7, 8),
                    (9, 10), (11, 12), (13, 14), (15, 16)
                ]
                for left, right in left_right_pairs:
                    kpts[[left, right]] = kpts[[right, left]]
            kpts[:, 0] = (kpts[:, 0] - params.crop_j) * scale_x
            kpts[:, 1] = (kpts[:, 1] - params.crop_i) * scale_y
            # set keypoints that are out of bounds to invisible
            for idx in range(kpts.shape[0]):
                x, y, v = kpts[idx]
                if x < 0 or x >= new_W or y < 0 or y >= new_H:
                    kpts[idx, 2] = 0
            new_keypoints.append(kpts.flatten().tolist())

        return new_keypoints
    
    def apply_params_rgb(self, img, params: SpatialTransformParams, dim: tuple):
        return self.apply_params_img(img, params, InterpolationMode.BILINEAR)
    
    def apply_params_depth(self, img, params: SpatialTransformParams, dim: tuple):
        return self.apply_params_img(img, params, InterpolationMode.NEAREST)
    
    def apply_params_semseg(self, img, params: SpatialTransformParams, dim: tuple):
        return self.apply_params_img(img, params, InterpolationMode.NEAREST)

    def apply_params_matseg(self, img, params: SpatialTransformParams, dim: tuple):
        return self.apply_params_img(img, params, InterpolationMode.NEAREST)

    def apply_params_partseg(self, img, params: SpatialTransformParams, dim: tuple):
        if img is None:
            return None
        if params.flip:
            img_np = np.array(img)
            # flip left-right parts for densepose partseg
            left_right_map = [(2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]
            for left, right in left_right_map:
                left_mask = img_np == left
                right_mask = img_np == right
                img_np[left_mask] = right
                img_np[right_mask] = left
            img = Image.fromarray(img_np)
        return self.apply_params_img(img, params, InterpolationMode.NEAREST)

    def apply_params_albedo(self, img, params: SpatialTransformParams, dim: tuple):
        return self.apply_params_img(img, params, InterpolationMode.BILINEAR)

    def apply_params_illum(self, img, params: SpatialTransformParams, dim: tuple):
        return self.apply_params_img(img, params, InterpolationMode.BILINEAR)

    def apply_params_normal(self, img, params: SpatialTransformParams, dim: tuple):
        """Apply transformation parameters to normal map."""
        if img is None:
            return None
        if params.flip:
            img_np = np.array(img)
            img_np[:,:,0] = 255 - img_np[:,:,0]  # flip x channel
            img = Image.fromarray(img_np)
        return self.apply_params_img(img, params, InterpolationMode.BILINEAR)

    def __call__(self, data):
        """Apply random transformations to image and mask pair."""
        if isinstance(data['rgb'], list):
            tmp_img = data['rgb'][0]
            is_video = True
        else:
            tmp_img = data['rgb']
            is_video = False

        params = self.generate_params(tmp_img)
        org_dim = tmp_img.size[::-1]  # (H, W)

        for k in data.keys():
            task = k.split('-')[0]
            func = getattr(self, f'apply_params_{task}')
            if is_video:
                data[k] = [func(frame, params, dim=org_dim) for frame in data[k]]
            else:
                data[k] = func(data[k], params, dim=org_dim)
        return data