import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedLatentLossFunction:
    """
    Unified Normals Loss Function for both video and image reconstruction estimation.
    This function handles both video and image reconstruction estimation.
    """

    def __init__(self, mse=None, huber=None, l1=None, cos=None, tgm=None):
        """ UnifiedLatentLossFunction constructor.
        """
        self.name = "UnifiedLatentLossFunction"
        self.mse = self._check_loss(mse)
        self.huber = self._check_loss(huber)
        self.l1 = self._check_loss(l1)
        self.cos = self._check_loss(cos)
        self.tgm = self._check_loss(tgm)

        if self.mse is not None:
            self.mse_loss = torch.nn.MSELoss()
        if self.huber is not None:
            self.huber_loss = torch.nn.SmoothL1Loss()
        if self.l1 is not None:
            self.l1_loss = torch.nn.L1Loss()
        if self.cos is not None:
            self.cos_sim = torch.nn.CosineSimilarity(dim=2)
        if self.tgm is not None:
            self.mse_per_element = torch.nn.MSELoss(reduction='none')

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

        pred, proj_name = current_estimate
        assert pred.ndim == 5, "expected shape: (B, T, C, H, W)."

        pred = pred.to(device=device, dtype=weight_dtype)
        gt = ground_truth[proj_name].to(device=device)
        is_video = ground_truth["is_video"]

        if self.mse is not None:
            loss['mse'] = self._check_nan(self._mse_loss(pred, gt))
            loss['total'] = loss['total'] + loss['mse']
        if self.huber is not None:
            loss['huber'] = self._check_nan(self._huber_loss(pred, gt))
            loss['total'] = loss['total'] + loss['huber']
        if self.l1 is not None:
            loss['l1'] = self._check_nan(self._l1_loss(pred, gt))
            loss['total'] = loss['total'] + loss['l1']
        if self.cos is not None:
            loss['cos'] = self._check_nan(self._cos_loss(pred, gt))
            loss['total'] = loss['total'] + loss['cos']
        if self.tgm is not None and is_video:
            loss['tgm'] = self._check_nan(self._tgm_loss(pred, gt))
            loss['total'] = loss['total'] + loss['tgm']
        return loss

    def _mse_loss(self, prediction, ground_truth):
        return self.mse["weight"] * self.mse_loss(prediction, ground_truth)

    def _huber_loss(self, prediction, ground_truth):
        return self.huber["weight"] * self.huber_loss(prediction, ground_truth)
    
    def _l1_loss(self, prediction, ground_truth):
        return self.l1["weight"] * self.l1_loss(prediction, ground_truth)

    def _cos_loss(self, prediction, ground_truth):
        cos_loss = 1 - self.cos_sim(prediction, ground_truth).mean()
        return self.cos["weight"] * cos_loss

    def _tgm_loss(self, prediction, ground_truth):
        """ Temporal Gradient Magnitude Loss
        """
        b, t, c, h, w = prediction.shape
        pred_grad = prediction[:, 1:, :, :, :] - prediction[:, :-1, :, :, :]
        gt_grad = ground_truth[:, 1:, :, :, :] - ground_truth[:, :-1, :, :, :]
        tgm_loss = self.mse_per_element(pred_grad, gt_grad)
        if self.tgm["max_diff"] is None:
            return self.tgm["weight"] * tgm_loss.mean()
        else:
            gt_grad_mag = torch.sqrt(torch.sum(gt_grad ** 2, dim=2, keepdim=True))  # (B, T-1, 1, H, W)
            mask = gt_grad_mag < self.tgm["max_diff"]
            return self.tgm["weight"] * (tgm_loss * mask).sum() / (c * mask.sum() + 1e-6)

    def _check_nan(self, tensor):
        """ Check if the tensor contains NaN values.
        """
        if not torch.isnan(tensor).any():
            return tensor
        else:
            print(f'NAN detected in tensor.')
            return torch.zeros_like(tensor)