# @ GonzaloMartinGarcia
# Task specific loss functions are from Depth Anything https://github.com/LiheYoung/Depth-Anything.
# Modifications have been made to improve numerical stability for this project (marked by '# add').

import torch
import torch.nn as nn
import torch.nn.functional as F

##################
# Loss Function
##################

class DepthLossFunction:
    def __init__(self):
        self.ssi_loss = ScaleAndShiftInvariantLoss()

    def __call__(self, current_estimate, ground_truth, device, weight_dtype):
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Return zero if there is no estimation
        if len(current_estimate) == 0:
            return loss

        # Post-process predicted image
        current_estimate = current_estimate.mean(dim=1, keepdim=True)
        current_estimate = torch.clamp(current_estimate, -1, 1)

        # Move ground truth and mask to proper device/dtype
        metric_depth = ground_truth["metric_depth"].to(device=device, dtype=weight_dtype)
        val_mask = ground_truth["valid_mask"].to(device=device, dtype=torch.bool)

        # Compute loss
        estimation_loss_ssi = self.ssi_loss(current_estimate, metric_depth, val_mask)
        if not torch.isnan(estimation_loss_ssi).any():
            loss = loss + estimation_loss_ssi
        else:
            print(f'[device: {device}] estimation_loss_ssi NAN: {estimation_loss_ssi}, is_input_nan: {torch.isnan(current_estimate).any()}, mean_input: {current_estimate.mean()}')
        return loss

##################
# Helper Functions
##################

# Scale and Shift Invariant Loss
class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask):
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        prediction, target = prediction.squeeze(1), target.squeeze(1)
        # add
        with torch.autocast(device_type='cuda', enabled=False):
            prediction = prediction.float()
            target     = target.float()

            scale, shift = compute_scale_and_shift_masked(prediction, target, mask)
            scaled_prediction = torch.abs(scale).view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
            if torch.any(mask):
                loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
            else:
                loss = nn.functional.l1_loss(scaled_prediction, target) * 0.0
        return loss
    
def compute_scale_and_shift_masked(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0 #1e-3
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    return x_0, x_1



