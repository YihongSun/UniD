import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedSemSegLossFunction:
    """
    Unified Normals Loss Function for both video and image normals estimation.
    This function handles both video and image normals estimation.
    """

    def __init__(self, l1=None, cw_l1=None, ce=None, dice=None, v_tgm=None, non_empty_reg=None):
        """ UnifiedSemSegLossFunction constructor.
        """
        self.name = "UnifiedSemSegLossFunction"
        self.l1 = self._check_loss(l1)
        self.cw_l1 = self._check_loss(cw_l1)
        self.ce = self._check_loss(ce)
        self.dice = self._check_loss(dice)
        self.v_tgm = self._check_loss(v_tgm)
        self.non_empty_reg = self._check_loss(non_empty_reg)

        if self.l1 is not None or self.cw_l1 is not None:
            self.l1_loss = torch.nn.SmoothL1Loss(reduction='none')
        
        if self.ce is not None:
            self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

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
            pred, pred_raw, proj_name = current_estimate
            pred_raw = pred_raw.to(device=device, dtype=weight_dtype)
        else:
            pred, pred_raw = current_estimate, None
        assert pred.ndim == 5, "expected shape: (B, T, C, H, W)."
        pred = pred.to(device=device, dtype=weight_dtype)
        mask = ground_truth[proj_name].to(device=device).long()     # B, T, H, W
        is_video = ground_truth["is_video"]
        # gt = ground_truth['anno'].to(device=device, dtype=weight_dtype)
        # if self.l1 is not None:
        #     loss['l1'] = self._check_nan(self._l1_loss(pred, gt, mask))
        #     loss['total'] = loss['total'] + loss['l1'] 
        # if self.cw_l1 is not None:
        #     class_weight = ground_truth['cat_weight'].to(device=device, dtype=weight_dtype)  # C
        #     loss['cw_l1'] = self._check_nan(self._cw_l1_loss(pred, gt, class_weight))
        #     loss['total'] = loss['total'] + loss['cw_l1']
        if self.dice is not None:
            pred_softmax = F.softmax(pred, dim=2)  # b, t, c, h, w
        if self.ce is not None:
            loss['ce'] = self._check_nan(self._ce_loss(pred, mask))
            loss['total'] = loss['total'] + loss['ce']
        if self.dice is not None:
            loss['dice'] = self._check_nan(self._dice_loss(pred_softmax, mask))
            loss['total'] = loss['total'] + loss['dice']
        if self.non_empty_reg is not None:
            loss['non_empty_reg'] = self._check_nan(self._non_empty_reg(pred, mask))
            loss['total'] = loss['total'] + loss['non_empty_reg']
        if is_video and self.v_tgm is not None and pred_raw is not None:
            loss['v_tgm'] = self._check_nan(self._v_tgm_loss(pred_raw, mask))
            loss['total'] = loss['total'] + loss['v_tgm']

        return loss

    def _l1_loss(self, pred, gt, mask):
        """ Compute the L1 loss
        """
        valid_mask = (mask > 0).type(pred.dtype)  # B, T, H, W
        l1_loss = self.l1_loss(pred, gt).mean(2) # B, T, H, W
        return self.l1["weight"] * (l1_loss * valid_mask).mean()
    
    def _cw_l1_loss(self, pred, gt, class_weight):
        """ Compute the class-weighted L1 loss
        """
        l1_loss = self.l1_loss(pred, gt).mean(2) # B, T, H, W
        cw_l1_loss = l1_loss * class_weight  # B, T, H, W
        return self.cw_l1["weight"] * cw_l1_loss.mean()
    
    def _ce_loss(self, pred, mask):
        """ Compute the cross-entropy loss
        """
        (b, t, c, h, w) = pred.shape
        pred = pred.reshape(b*t, c, h, w)
        mask = mask.reshape(b*t, h, w)

        loss = self.ce_loss(pred, mask)  # b*t, h, w
        ignore_index = self.ce.get("ignore_index", None)
        if ignore_index is not None:
            loss = loss * (mask != ignore_index).type(loss.dtype)
        return self.ce["weight"] * loss.mean()

    def _dice_loss(self, pred_softmax, mask):
        """ Compute the Dice loss
        Args:
            pred_softmax: predictions of shape (B, T, C, H, W) after softmax
            mask: ground truth labels of shape (B, T, H, W)
        Returns:
            Dice loss value (1 - Dice coefficient)
        """
        (b, t, c, h, w) = pred_softmax.shape
        pred_softmax = pred_softmax.reshape(b*t, c, h, w)
        mask = mask.reshape(b*t, h, w)
        dice_smooth = self.dice.get("smooth", 1.0)
        
        # Convert mask to one-hot encoding
        mask_one_hot = F.one_hot(mask, num_classes=c)  # b*t, h, w, c
        mask_one_hot = mask_one_hot.permute(0, 3, 1, 2).float()  # b*t, c, h, w
        
        # Handle ignore index if specified
        ignore_index = self.dice.get("ignore_index", None)
        if ignore_index is not None:
            valid_mask = (mask != ignore_index).float()  # b*t, h, w
            valid_mask = valid_mask.unsqueeze(1)  # b*t, 1, h, w
        else:
            valid_mask = torch.ones_like(mask, dtype=pred_softmax.dtype).unsqueeze(1)
        
        # Apply valid mask to predictions and targets
        pred_softmax = pred_softmax * valid_mask
        mask_one_hot = mask_one_hot * valid_mask
        
        # Flatten spatial dimensions
        pred_flat = pred_softmax.reshape(b*t, c, -1)  # b*t, c, h*w
        mask_flat = mask_one_hot.reshape(b*t, c, -1)  # b*t, c, h*w
        
        # Compute Dice coefficient for each class
        intersection = (pred_flat * mask_flat).sum(dim=2)  # b*t, c
        
        if self.dice.get("squared", False):
            # Squared Dice variant: more stable gradients
            pred_sum = (pred_flat ** 2).sum(dim=2)  # b*t, c
            mask_sum = (mask_flat ** 2).sum(dim=2)  # b*t, c
        else:
            pred_sum = pred_flat.sum(dim=2)  # b*t, c
            mask_sum = mask_flat.sum(dim=2)  # b*t, c
        
        dice_coeff = (2.0 * intersection + dice_smooth) / (pred_sum + mask_sum + dice_smooth)
        
        # Dice loss = 1 - Dice coefficient
        dice_loss = 1.0 - dice_coeff
        
        # Average over classes and batch
        dice_loss = dice_loss.mean()
        
        return self.dice["weight"] * dice_loss
    
    def _non_empty_reg(self, pred, mask):
        """ Compute the non-empty region regularization loss
        """
        pred_probs = F.softmax(pred, dim=2)   # B, T, C, H, W
        empty_prob = pred_probs[:, :, self.non_empty_reg['empty_index'], :, :]  # B, T, H, W
        loss = -torch.log(1.0 - empty_prob + 1e-4)  # B, T, H, W
        non_empty_loss = loss * (mask == self.non_empty_reg['pos_index']).type(loss.dtype)
        return self.non_empty_reg["weight"] * non_empty_loss.mean()
    
    def _v_tgm_loss(self, pred_raw, mask):
        """ Compute the temporal gradient magnitude loss
        """
        (b, t, c, h, w) = pred_raw.shape
        pred_tg = pred_raw[:, 1:] - pred_raw[:, :-1]    # B, T-1, C, H, W
        tgt_mask = torch.all(mask[:,:1] == mask, dim=1, keepdims=True) & (mask[:,:1] > 0)  # B, 1, H, W
        
        if tgt_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_raw.device)

        pred_tg2reg = pred_tg[tgt_mask[:,:,None].repeat(1, t-1, c, 1, 1)]
        targ_tg = torch.zeros_like(pred_tg2reg)
        v_tgm_loss = F.l1_loss(pred_tg2reg, targ_tg)

        return self.v_tgm["weight"] * v_tgm_loss

    def _check_nan(self, tensor):
        """ Check if the tensor contains NaN values.
        """
        if not torch.isnan(tensor).any():
            return tensor
        else:
            breakpoint()
            print(f'NAN detected in tensor.')
            return torch.zeros_like(tensor)