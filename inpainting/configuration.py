import os
from argparse import Namespace
from warnings import warn

import torch
from pkg_resources import parse_version
from pytorch_lightning.logging import TensorBoardLogger, rank_zero_only
from torch.utils.tensorboard import SummaryWriter

model_path = 'models'


class MyLogger(TensorBoardLogger):

    def __init__(self, model_name):
        super(MyLogger, self).__init__(model_path, model_name)
        self.train_experiment = self.get_summary_writer('train')
        self.val_experiment = self.get_summary_writer('val')

    def get_summary_writer(self, suffix=''):
        root_dir = os.path.join(self.save_dir, self.name)
        os.makedirs(root_dir, exist_ok=True)
        log_dir = os.path.join(root_dir, "version_" + str(self.version) + '/' + suffix)
        return SummaryWriter(log_dir=log_dir, **self.kwargs)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, dict):
                if 'train' in v:
                    self.train_experiment.add_scalar(k, v['train'], step)
                if 'val' in v:
                    self.val_experiment.add_scalar(k, v['val'], step)
            else:
                self.experiment.add_scalar(k, v, step)

    @rank_zero_only
    def log_hyperparams(self, params):
        if params is None:
            return

        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)
        params = dict(params)

        if parse_version(torch.__version__) < parse_version("1.3.0"):
            warn(
                f"Hyperparameter logging is not available for Torch version {torch.__version__}."
                " Skipping log_hyperparams. Upgrade to Torch 1.3.0 or above to enable"
                " hyperparameter logging."
            )
        else:
            from torch.utils.tensorboard.summary import hparams
            exp, ssi, sei = hparams(params, {})
            writer = self.experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)
        # some alternative should be added
        self.tags.update(params)
