from .metric_base import MaskedMetric
from .ops import mape
from torch.nn import functional as F
import torch

from torchmetrics.utilities.checks import _check_same_shape

epsilon = 1e-6

class MaskedMAE(MaskedMetric):
    def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 compute_on_step=False,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedMAE, self).__init__(metric_fn=F.l1_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        compute_on_step=compute_on_step,
                                        dist_sync_on_step=dist_sync_on_step,
                                        process_group=process_group,
                                        dist_sync_fn=dist_sync_fn,
                                        metric_kwargs={'reduction': 'none'},
                                        at=at)


class MaskedMAPE(MaskedMetric):
    def __init__(self,
                 mask_nans=False,
                 compute_on_step=False,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedMAPE, self).__init__(metric_fn=mape,
                                         mask_nans=mask_nans,
                                         mask_inf=True,
                                         compute_on_step=compute_on_step,
                                         dist_sync_on_step=dist_sync_on_step,
                                         process_group=process_group,
                                         dist_sync_fn=dist_sync_fn,
                                         at=at)


class MaskedMSE(MaskedMetric):
    def __init__(self,
                 mask_nans=False,
                 compute_on_step=False,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedMSE, self).__init__(metric_fn=F.mse_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=True,
                                        compute_on_step=compute_on_step,
                                        dist_sync_on_step=dist_sync_on_step,
                                        process_group=process_group,
                                        dist_sync_fn=dist_sync_fn,
                                        metric_kwargs={'reduction': 'none'},
                                        at=at)


class MaskedMRE(MaskedMetric):
    def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 compute_on_step=False,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedMRE, self).__init__(metric_fn=F.l1_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        compute_on_step=compute_on_step,
                                        dist_sync_on_step=dist_sync_on_step,
                                        process_group=process_group,
                                        dist_sync_fn=dist_sync_fn,
                                        metric_kwargs={'reduction': 'none'},
                                        at=at)
        self.add_state('tot', dist_reduce_fx='sum', default=torch.tensor(0., dtype=torch.float))

    def _compute_masked(self, y_hat, y, mask):
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        mask = self._check_mask(mask, val)
        val = torch.where(mask, val, torch.tensor(0., device=y.device, dtype=torch.float))
        y_masked = torch.where(mask, y, torch.tensor(0., device=y.device, dtype=torch.float))
        return val.sum(), mask.sum(), y_masked.sum()

    def _compute_std(self, y_hat, y):
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        return val.sum(), val.numel(), y.sum()

    def compute(self):
        if self.tot > epsilon:
            res = self.value / self.tot
        else:
            res = self.value
        self.clear()
        return res

    def update(self, y_hat, y, mask=None):
        y_hat = y_hat[:, self.at]
        y = y[:, self.at]
        if mask is not None:
            mask = mask[:, self.at]
        if self.is_masked(mask):
            val, numel, tot = self._compute_masked(y_hat, y, mask)
        else:
            val, numel, tot = self._compute_std(y_hat, y)
        self.value += val
        self.numel += numel
        self.tot += tot

