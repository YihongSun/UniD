import torch
import torch.nn as nn
import torch.nn.functional as F
from .normals_loss import AngularLoss

class UnifiedNormalsLossFunction:
    """
    Unified Normals Loss Function for both video and image normals estimation.
    This function handles both video and image normals estimation.
    """

    def __init__(self, i_ang=None, v_ang=None, far_p=None):
        """ UnifiedNormalsLossFunction constructor.
        """
        self.name = "UnifiedNormalsLossFunction"
        self.i_ang = self._check_loss(i_ang)
        self.v_ang = self._check_loss(v_ang)
        self.far_p = self._check_loss(far_p)

        if self.i_ang is not None or self.v_ang is not None:
            self.ang_loss = AngularLoss()

    def _check_loss(self, l):
        return l if l is not None and l["weight"] > 0 else None

    def __call__(self, current_estimate, ground_truth, device, weight_dtype):
        """ Compute the unified normals loss for the current estimate.
        Args:
            current_estimate (torch.Tensor): The current normals estimate, shape (B, T, C, H, W).
            ground_truth (dict): A dictionary containing ground truth normals and masks.
            device (torch.device): The device to perform computations on.
            weight_dtype (torch.dtype): The dtype for weights.
        Returns:
            torch.Tensor: The computed loss.
        """
        loss = {'total': torch.tensor(0.0, device=device, requires_grad=True)}

        # Return zero if there is no estimation
        if len(current_estimate) == 0:
            return loss

        if type(current_estimate) is tuple:
            current_estimate = current_estimate[0]

        # Normalize prediction
        assert current_estimate.ndim == 5, "expected shape: (B, T, C, H, W)."
        norm = torch.norm(current_estimate, p=2, dim=2, keepdim=True) + 1e-5
        current_estimate = current_estimate / norm
        current_estimate = torch.clamp(current_estimate, -1, 1)

        # Move ground truth and mask to proper device/dtype
        normals_gt = ground_truth["normal"].to(device=device, dtype=weight_dtype)
        val_mask = ground_truth["normal_valid_mask"].to(device=device, dtype=torch.bool)
        is_video = ground_truth["is_video"]

        if self.i_ang is not None:
            loss['i_ang'] = self._check_nan(self._i_ang_loss(current_estimate, normals_gt, val_mask))
            loss['total'] = loss['total'] + loss['i_ang'] 
        if self.v_ang is not None:
            loss['v_ang'] = self._check_nan(self._v_ang_loss(current_estimate, normals_gt, val_mask))
            loss['total'] = loss['total'] + loss['v_ang']
        if self.far_p is not None:
            far_mask = ground_truth["depth_invalid_far_mask"].to(device=device, dtype=torch.bool)
            loss['far_p'] = self._check_nan(self._far_p_loss(current_estimate, far_mask))
            loss['total'] = loss['total'] + loss['far_p']

        return loss

    def _i_ang_loss(self, prediction, normals_gt, val_mask):
        B, T, C, H, W = prediction.shape
        pred = prediction.view(B*T, C, H, W)
        gt = normals_gt.view(B*T, C, H, W)
        mask = val_mask.view(B*T, 1, H, W)
        return self.i_ang["weight"] * self.ang_loss(pred, gt, mask)

    def _v_ang_loss(self, prediction, normals_gt, val_mask):
        B, T, C, H, W = prediction.shape
        pred = prediction.view(B*T, C, H, W)
        gt = normals_gt.view(B*T, C, H, W)
        mask = val_mask.view(B*T, 1, H, W)
        return self.v_ang["weight"] * self.ang_loss(pred, gt, mask)

    def _far_p_loss(self, prediction, far_mask):
        """ Encourage -1 prediction for regions beyond the far plane.
            prediction :    (B, T, 3, H, W)
            far_mask :      (B, T, 1, H, W)
        """
        (B, T, C, H, W) = prediction.shape
        pred = prediction.view(B*T, C, H, W)
        far_targ = torch.ones_like(pred)
        far_targ[:,0] = 0; far_targ[:,1] = 0      # Set normal to point towards camera
        far_mask = far_mask.view(B*T, 1, H, W).to(dtype=torch.bool)
        return self.far_p["weight"] * self.ang_loss(pred, far_targ, far_mask)

    def _check_nan(self, tensor):
        """ Check if the tensor contains NaN values.
        """
        if not torch.isnan(tensor).any():
            return tensor
        else:
            print(f'NAN detected in tensor.')
            return torch.zeros_like(tensor)