import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaClinix import nnUNetTrainerMambaClinix
from nnunetv2.training.loss.mambaclinix_loss import MambaClinix_compound_losses
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
import numpy as np

class nnUNetTrainerMambaClinixTRS(nnUNetTrainerMambaClinix):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = MambaClinix_compound_losses(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5,
             'do_bg': False, 'ddp': self.is_ddp},
            {},
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5,
             'do_bg': False, 'alpha': 0.3, 'beta': 0.4,
             'num_region_per_axis': (8, 8, 14)},
            weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerMambaClinixTRS_500epochs(nnUNetTrainerMambaClinixTRS):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
