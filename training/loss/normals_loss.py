# @ GonzaloMartinGarcia
# Task specific loss functions are from Depth Anything https://github.com/LiheYoung/Depth-Anything.
# Modifications have been made to improve numerical stability for this project (marked by '# add').

import torch
import torch.nn as nn
import torch.nn.functional as F

##################
# Loss Functions
##################

class NormalsLossFunction:
    def __init__(self):
        self.angular_loss_norm = AngularLoss()

    def __call__(self, current_estimate, ground_truth, device, weight_dtype):
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Return zero if there is no estimation
        if len(current_estimate) == 0:
            return loss

        # Normalize prediction
        norm = torch.norm(current_estimate, p=2, dim=1, keepdim=True) + 1e-5
        current_estimate = current_estimate / norm
        current_estimate = torch.clamp(current_estimate, -1, 1)

        # Move ground truth and mask to proper device/dtype
        normals = ground_truth["normals"].to(device=device, dtype=weight_dtype)
        val_mask = ground_truth["valid_mask"].to(device=device, dtype=torch.bool)

        # Compute angular loss
        estimation_loss_ang_norm = self.angular_loss_norm(current_estimate, normals, val_mask)
        if not torch.isnan(estimation_loss_ang_norm).any():
            loss = loss + estimation_loss_ang_norm
        else:
            print(f'[device: {device}] estimation_loss_ang_norm NAN: {estimation_loss_ang_norm}, is_input_nan: {torch.isnan(current_estimate).any()}, mean_input: {current_estimate.mean()}')

        return loss

##################
# Helper Functions
##################

# Angluar Loss
class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()
        self.name = "Angular"

    def forward(self, prediction, target, mask=None):
        with torch.autocast(device_type='cuda', enabled=False):
            prediction = prediction.float()
            target = target.float()
            mask   = mask[:,0,:,:]    
            dot_product = torch.sum(prediction * target, dim=1)
            dot_product = torch.clamp(dot_product, -1.0 + 1e-6, 1.0 - 1e-6)
            angle = torch.acos(dot_product)
            if torch.any(mask):
                angle = angle[mask]
            else:
                angle = angle * 0.0
            loss = angle.mean()
        return loss