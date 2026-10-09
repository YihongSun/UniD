# @ GonzaloMartinGarcia
# Task specific loss functions are from Depth Anything https://github.com/LiheYoung/Depth-Anything.
# Modifications have been made to improve numerical stability for this project (marked by '# add').

import torch
import torch.nn as nn
import torch.nn.functional as F
from .depth_loss import ScaleAndShiftInvariantLoss, compute_scale_and_shift_masked

##################
# Loss Function
##################

class UnifiedDepthLossFunction:
    """
    Unified Depth Loss Function for both video and image depth estimation.
    This function handles both video and image depth estimation.
    """
    def __init__(self, i_ssi=None, v_ssi=None, v_flow=None, v_tgm=None, far_p=None):
        """ UnifiedDepthLossFunction constructor.
        """
        self.name = "UnifiedDepthLossFunction"
        self.i_ssi = self._check_loss(i_ssi)
        self.v_ssi = self._check_loss(v_ssi)
        self.v_flow = self._check_loss(v_flow)
        self.v_tgm = self._check_loss(v_tgm)
        self.far_p = self._check_loss(far_p)

        if self.i_ssi is not None or self.v_ssi is not None:
            self.ssi_loss = ScaleAndShiftInvariantLoss()
        if self.v_flow is not None:
            self.flow_loss = FlowConsistencyLoss(cycle_consistency_thrd=self.v_flow["cycle_cons_thrd"])
        
    def _check_loss(self, l):
        return l if l is not None and l["weight"] > 0 else None

    def __call__(self, current_estimate, batch, device, weight_dtype):
        """ Compute the unified depth loss for the current estimate.
        Args:
            current_estimate (torch.Tensor): The current depth estimate, shape (B, T, C, H, W).
            batch (dict): A dictionary containing ground truth and masks.
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
        
        # Post-process predicted image
        assert current_estimate.ndim == 5, "expected shape: (B, T, C, H, W)."
        prediction = current_estimate.mean(dim=2, keepdim=True)     # (B, T, 1, H, W)
        prediction = torch.clamp(prediction, -1, 1)

        # Move ground truth and mask to proper device/dtype
        metric_depth = batch["depth"].to(device=device, dtype=weight_dtype)
        val_mask = batch["depth_valid_mask"].to(device=device, dtype=torch.bool)
        is_video = batch["is_video"]

        if self.i_ssi is not None:
            loss['i_ssi'] = self._check_nan(self._i_ssi_loss(prediction, metric_depth, val_mask))
            loss['total'] = loss['total'] + loss['i_ssi']
        if is_video and self.v_ssi is not None:
            loss['v_ssi'] = self._check_nan(self._v_ssi_loss(prediction, metric_depth, val_mask))
            loss['total'] = loss['total'] + loss['v_ssi']
        if is_video and self.v_tgm is not None:
            loss['v_tgm'] = self._check_nan(self._v_tgm_loss(prediction, metric_depth, val_mask))
            loss['total'] = loss['total'] + loss['v_tgm']
        if self.v_flow is not None:
            loss['v_flow'] = self._check_nan(self._v_flow_loss(prediction, batch, val_mask))
            loss['total'] = loss['total'] + loss['v_flow']
        if self.far_p is not None:
            far_mask = batch["depth_invalid_far_mask"].to(device=device, dtype=torch.bool)
            loss['far_p'] = self._check_nan(self._far_p_loss(prediction, far_mask))
            loss['total'] = loss['total'] + loss['far_p']

        return loss


    def _i_ssi_loss(self, prediction, metric_depth, val_mask):
        """ Compute the image depth estimation loss.
        """

        # Combine the batch and time dimensions for image depth estimation.
        B, T, _, H, W = prediction.shape
        p = prediction.view(B*T, 1, H, W)
        md = metric_depth.view(B*T, 1, H, W)
        v = val_mask.view(B*T, 1, H, W)

        return self.i_ssi["weight"] * self.ssi_loss(p, md, v)


    def _v_ssi_loss(self, prediction, metric_depth, val_mask):
        """ Compute the image depth estimation loss.
        """

        # Concatenate along time dimension to force video-wise shift/scale computation.
        B, T, _, H, W = prediction.shape
        p = torch.cat([prediction[:,t] for t in range(T)], dim=2) # (B, 1, T*H, W)
        md = torch.cat([metric_depth[:,t] for t in range(T)], dim=2) # (B, 1, T*H, W)
        v = torch.cat([val_mask[:,t] for t in range(T)], dim=2) # (B, 1, T*H, W)

        return self.v_ssi["weight"] * self.ssi_loss(p, md, v)

    def _far_p_loss(self, prediction, far_mask):
        """ Encourage -1 prediction for regions beyond the far plane.
        """

        far_pred = prediction[far_mask]
        far_targ = torch.ones_like(far_pred)
        if far_pred.numel() > 0:
            far_p_loss = F.l1_loss(far_pred, far_targ) * far_mask.sum() / far_mask.numel()
        else:
            far_p_loss = torch.tensor(0.0, device=far_mask.device)

        return self.far_p["weight"] * far_p_loss
    
    def _v_flow_loss(self, prediction, batch, val_mask):
        """ Compute the flow consistency loss.
        """
        return self.flow_loss(prediction, batch, val_mask)
    
    def _v_tgm_loss(self, prediction, metric_depth, val_mask):
        """ Compute the temporal gradient loss.
        """

        # Concatenate along time dimension to force video-wise shift/scale computation.
        B, T, _, H, W = prediction.shape
        pred = prediction.view(B, T*H, W)
        targ = metric_depth.view(B, T*H, W)
        mask = val_mask.view(B, T*H, W)

        scale, shift = compute_scale_and_shift_masked(pred, targ, mask)
        ss_pred = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)

        ss_pred = ss_pred.view(B, T, H, W)
        targ = targ.view(B, T, H, W)
        mask = mask.view(B, T, H, W)

        pred_tg = ss_pred[:, 1:] - ss_pred[:, :-1]   # (B, T-1, H, W)
        targ_tg = targ[:, 1:] - targ[:, :-1]         # (B, T-1, H, W)
        mask_tg = mask[:, 1:] & mask[:, :-1] & (torch.abs(targ_tg) < self.v_tgm["tg_thrd"])

        loss = F.l1_loss(pred_tg[mask_tg], targ_tg[mask_tg])
        return self.v_tgm["weight"] * loss

    
    def _check_nan(self, tensor):
        """ Check if the tensor contains NaN values.
        """
        if not torch.isnan(tensor).any():
            return tensor
        else:
            print(f'NAN detected in tensor.')
            return torch.zeros_like(tensor)


##################
# Helper Functions
##################

# Flow Consistency Loss
class FlowConsistencyLoss(nn.Module):
    def __init__(self, cycle_consistency_thrd=0.34):
        super().__init__()
        self.name = "FLowLoss"
        self.cycle_cons_thrd = cycle_consistency_thrd

    def forward(self, prediction, batch, mask):
        prediction = prediction / torch.std(prediction) # normalize for training stability - need to try teacher
        
        pred_prev = prediction[:, :-1] # (B, T-1, 1, H, W)
        pred_next = prediction[:, 1:]  # (B, T-1, 1, H, W)

        loss = self._one_dir_loss(
            pred_curr=pred_prev,
            pred_src=pred_next,
            sample_src=batch["flow/sample_next"],
            error_curr=batch["flow/error_prev"]
        ) / 2.0

        loss = loss + self._one_dir_loss(
            pred_curr=pred_next,
            pred_src=pred_prev,
            sample_src=batch["flow/sample_prev"],
            error_curr=batch["flow/error_next"]
        ) / 2.0

        return loss

    def _one_dir_loss(self, pred_curr, pred_src, sample_src, error_curr):
        B, t, _, H, W = pred_curr.shape
        warped_pred_src = F.grid_sample(
            pred_src.reshape(-1, 1, H, W),
            sample_src.reshape(-1, H, W, 2),
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).reshape(B, t, 1, H, W)

        cycle_mask = error_curr[:,:,None] < self.cycle_cons_thrd
        if cycle_mask.any():
            cycle_loss = (pred_curr[cycle_mask] - warped_pred_src[cycle_mask]).abs().sum() / pred_curr.numel()
        else:
            cycle_loss = 0

        return cycle_loss