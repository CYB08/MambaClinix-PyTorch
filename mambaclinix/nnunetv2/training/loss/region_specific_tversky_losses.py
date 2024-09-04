import torch
from typing import Callable
from torch import nn


class RegionSpecificLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, smooth=1e-5, num_region_per_axis=(16, 16, 16),
                 do_bg=False, batch_dice=True, alpha=0.3, beta=0.4):
        """
        Region-specific Tversky loss function
        """
        super(RegionSpecificLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.do_bg = do_bg
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.dim = len(num_region_per_axis)

        assert self.dim in [2, 3], "The num of dim must be 2 or 3."
        if self.dim == 3:
            self.pool = nn.AdaptiveAvgPool3d(num_region_per_axis)
        elif self.dim == 2:
            self.pool = nn.AdaptiveAvgPool2d(num_region_per_axis)


    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape
        assert self.dim == (len(shp_x) - 2), "The region size must match the data's size."

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        region_tp, region_fp, region_fn, _ = self.get_region_tp_fp_fn(x, y, self.batch_dice, loss_mask)

        alpha = self.alpha + self.beta * (region_fp + self.smooth) / (region_fp + region_fn + self.smooth)
        beta = self.alpha + self.beta * (region_fn + self.smooth) / (region_fp + region_fn + self.smooth)

        region_tversky = (region_tp + self.smooth) / (region_tp + alpha * region_fp + beta * region_fn + self.smooth)

        if self.batch_dice:
            region_tversky = region_tversky.mean(list(range(1, len(shp_x) - 1)))
        else:
            region_tversky = region_tversky.mean(list(range(2, len(shp_x))))

        if not self.do_bg:
            if self.batch_dice:
                region_tversky = region_tversky[1:]
            else:
                region_tversky = region_tversky[:, 1:]

        region_tversky = region_tversky.mean()

        return -region_tversky


    def get_region_tp_fp_fn(self, net_output, gt, batch_dice=None, mask=None):

        with torch.no_grad():
            if net_output.ndim != gt.ndim:
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if net_output.shape == gt.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                y_onehot = torch.zeros(net_output.shape, device=net_output.device)
                y_onehot.scatter_(1, gt.long(), 1)

        tp = net_output * y_onehot
        fp = net_output * (1 - y_onehot)
        fn = (1 - net_output) * y_onehot
        tn = (1 - net_output) * (1 - y_onehot)

        if mask is not None:
            with torch.no_grad():
                mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
            tp *= mask_here
            fp *= mask_here
            fn *= mask_here

        # region specific pooling
        tp = self.pool(tp)
        fp = self.pool(fp)
        fn = self.pool(fn)

        # batch-scale integration
        if batch_dice:
            tp = tp.sum(dim=0, keepdim=False)
            fp = fp.sum(dim=0, keepdim=False)
            fn = fn.sum(dim=0, keepdim=False)

        return tp, fp, fn, tn