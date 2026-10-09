import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedImageLossFunction:
    """
    Unified Normals Loss Function for both video and image reconstruction estimation.
    This function handles both video and image reconstruction estimation.
    """

    def __init__(self, mse=None):
        """ UnifiedImageLossFunction constructor.
        """
        self.name = "UnifiedImageLossFunction"
        self.mse = self._check_loss(mse)

        if self.mse is not None:
            self.mse_loss = torch.nn.MSELoss()

    def _check_loss(self, l):
        return l if l is not None and l["weight"] > 0 else None

    def __call__(self, current_estimate, ground_truth, device, weight_dtype):
        """ Compute the unified image reconstruction loss for the current estimate.
        Args:
            current_estimate (torch.Tensor): The current estimate, shape (B, T, C, H, W).
            ground_truth (dict): A dictionary containing ground truth reconstruction targets.
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
            pred, pred_raw, proj_name = current_estimate
            pred_raw = pred_raw.to(device=device, dtype=weight_dtype)
        else:
            pred, pred_raw, proj_name = current_estimate, None, None
        assert pred.ndim == 5, "expected shape: (B, T, C, H, W)."

        pred = torch.clamp(pred, -1, 1)
        pred = pred.to(device=device, dtype=weight_dtype)
        gt = ground_truth[proj_name].to(device=device)
        is_video = ground_truth["is_video"]

        if self.mse is not None:
            loss['mse'] = self._check_nan(self._mse_loss(pred, gt))
            loss['total'] = loss['total'] + loss['mse'] 

        return loss

    def _mse_loss(self, prediction, ground_truth):
        return self.mse["weight"] * self.mse_loss(prediction, ground_truth)

    def _check_nan(self, tensor):
        """ Check if the tensor contains NaN values.
        """
        if not torch.isnan(tensor).any():
            return tensor
        else:
            print(f'NAN detected in tensor.')
            return torch.zeros_like(tensor)