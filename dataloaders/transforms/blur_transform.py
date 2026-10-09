import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode, RandomResizedCrop
import torchvision.transforms.functional as TF
from dataclasses import dataclass

@dataclass
class BlurTransformParams:
    """Container for transformation parameters that can be applied to multiple samples."""
    skip: bool
    motion_kernel: int
    blur_kernel: int
    motion_angle: int
    

class BlurTransform:
    def __init__(
        self,
        motion_kernel_range=(25,50),
        blur_kernel_range=(3,10),
        angle_range=(0, 360),
        p=0,
    ):
        self.motion_kernel_range = motion_kernel_range
        self.blur_kernel_range = blur_kernel_range
        self.angle_range = angle_range
        self.prob = p

    def generate_params(self) -> BlurTransformParams:
        """Generate transformation parameters for a given image."""
        
        if random.random() < self.prob:
            motion_kernel = random.randint(self.motion_kernel_range[0], self.motion_kernel_range[1])
            blur_kernel = random.randint(self.blur_kernel_range[0], self.blur_kernel_range[1])
            motion_angle = random.uniform(self.angle_range[0], self.angle_range[1])
            skip = False
        else:
            motion_kernel, blur_kernel, motion_angle = 0, 0, 0
            skip = True

        params = BlurTransformParams(
            skip=skip,
            motion_kernel=motion_kernel,
            blur_kernel=blur_kernel,
            motion_angle=motion_angle,
        )
        return params
    
    def gaussian_blur_kernel(self, size, sigma=None):
        """2D Gaussian kernel."""
        if size % 2 == 0:
            size += 1
        
        if sigma is None:
            sigma = size / 6.0
        
        ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / kernel.sum()
    
    def gaussian_motion_blur_kernel(self, size, angle, sigma=None):
        """Gaussian-weighted motion blur kernel."""
        if size % 2 == 0:
            size += 1
        
        if sigma is None:
            sigma = size / 6.0
        
        kernel_size = int(np.ceil(size * 1.5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = torch.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        half_size = size / 2.0
        t = torch.linspace(-half_size, half_size, size * 3)
        
        dx = torch.cos(angle)
        dy = torch.sin(angle)
        
        x = t * dx
        y = t * dy
        weights = torch.exp(-(t**2) / (2 * sigma**2))
        
        for i in range(len(t)):
            px = int(torch.round(center + x[i]).item())
            py = int(torch.round(center + y[i]).item())
            
            if 0 <= px < kernel_size and 0 <= py < kernel_size:
                kernel[py, px] += weights[i]
        
        return kernel / kernel.sum() if kernel.sum() > 0 else kernel
    
    def combined_blur_kernel(self, motion_length, motion_angle, motion_sigma=None, 
                         gaussian_size=None, gaussian_sigma=None):
        """Motion blur kernel convolved with Gaussian blur kernel."""
        motion_kernel = self.gaussian_motion_blur_kernel(
            motion_length, motion_angle, motion_sigma
        )
        
        if gaussian_size is None:
            return motion_kernel
        
        gaussian_kernel = self.gaussian_blur_kernel(gaussian_size, gaussian_sigma)
        
        # Convolve kernels
        padding = gaussian_kernel.shape[0] // 2
        motion_padded = F.pad(motion_kernel.unsqueeze(0).unsqueeze(0), 
                            (padding, padding, padding, padding))
        combined = F.conv2d(motion_padded, gaussian_kernel.unsqueeze(0).unsqueeze(0))
        combined = combined.squeeze()
        
        return combined / combined.sum()

    def apply_blur(self, img):
        params = self.generate_params()
        if params.skip:
            return img

        kernel = self.combined_blur_kernel(
            motion_length=params.motion_kernel,
            motion_angle=torch.deg2rad(torch.tensor(params.motion_angle)),
            motion_sigma=None,
            gaussian_size=params.blur_kernel,
            gaussian_sigma=None,
        ).unsqueeze(0).unsqueeze(0) # (1,1,H,W)
        kernel_size = kernel.shape[-1]
        
        # Apply convolution
        img = TF.to_tensor(img)
        C, H, W = img.shape
        kernel = kernel.repeat(C, 1, 1, 1)
        img_padded = F.pad(
            img[None], [kernel_size//2]*4, mode='replicate'
        )
        blurred = F.conv2d(
            img_padded, kernel, groups=3
        ).squeeze(0)

        return TF.to_pil_image(blurred)

    def __call__(self, data):
        """Apply random transformations to image and mask pair."""
        if isinstance(data['rgb'], list):
            blurred_input = [self.apply_blur(frame) for frame in data['rgb']]
        else:
            blurred_input = self.apply_blur(data['rgb'])

        data['input'] = blurred_input
        return data