import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedBoundaryLossFunction:
    """
    Unified Normals Loss Function for both video and image normals estimation.
    This function handles both video and image normals estimation.
    """

    def __init__(self, mse=None, pos_mse=None, neg_mse=None):
        """ UnifiedBoundaryLossFunction constructor.
        """
        self.name = "UnifiedBoundaryLossFunction"
        self.mse = self._check_loss(mse)
        self.pos_mse = self._check_loss(pos_mse)
        self.neg_mse = self._check_loss(neg_mse)

        if self.mse is not None or self.pos_mse is not None or self.neg_mse is not None:
            self.mse_loss = torch.nn.MSELoss(reduction='none')

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
            pred = current_estimate[0]
        else:
            pred = current_estimate
        assert pred.ndim == 5, "expected shape: (B, T, C, H, W)."
        pred = pred.to(device=device, dtype=weight_dtype)
        gt = ground_truth['boundary'].to(device=device, dtype=weight_dtype)
        val_mask = ground_truth["boundary_valid_mask"].to(device=device, dtype=torch.bool)

        if self.mse is not None:
            loss['mse'] = self._check_nan(self._mse_loss(pred, gt, val_mask))
            loss['total'] = loss['total'] + loss['mse']
        if self.pos_mse is not None:
            loss['pos_mse'] = self._check_nan(self._pos_mse_loss(pred, gt, val_mask, thrd=self.pos_mse["thrd"]))
            loss['total'] = loss['total'] + loss['pos_mse']
        if self.neg_mse is not None:
            loss['neg_mse'] = self._check_nan(self._neg_mse_loss(pred, gt, val_mask, thrd=self.neg_mse["thrd"]))
            loss['total'] = loss['total'] + loss['neg_mse']

        return loss

    def _mse_loss(self, pred, gt, mask):
        """ Compute the L1 loss
        """
        mse_loss = self.mse_loss(pred, gt) * mask.type(mse_loss.dtype)  # B, T, C, H, W
        return self.mse["weight"] * mse_loss.mean()
    
    def _pos_mse_loss(self, pred, gt, mask, thrd=0.01):
        """ Compute the normalized MSE loss
        """
        pos_mask = (gt > thrd) & mask  # B, T, C, H, W

        num_pos = pos_mask.sum().float()
        mse_loss = self.mse_loss(pred, gt)  # B, T, C, H, W

        pos_loss = (mse_loss * pos_mask.type(mse_loss.dtype)).sum() / (num_pos + 1e-6)

        return self.pos_mse["weight"] * pos_loss
    
    def _neg_mse_loss(self, pred, gt, mask, thrd=0.01):
        """ Compute the normalized MSE loss
        """
        neg_mask = (gt <= thrd) & mask  # B, T, C, H, W

        num_neg = neg_mask.sum().float()
        mse_loss = self.mse_loss(pred, gt)  # B, T, C, H, W

        neg_loss = (mse_loss * neg_mask.type(mse_loss.dtype)).sum() / (num_neg + 1e-6)

        return self.neg_mse["weight"] * neg_loss

    def _check_nan(self, tensor):
        """ Check if the tensor contains NaN values.
        """
        if not torch.isnan(tensor).any():
            return tensor
        else:
            print(f'NAN detected in tensor.')
            return torch.zeros_like(tensor)