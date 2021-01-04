import os,sys
import re
import math
from datetime import datetime
import time
from pathlib import Path

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar

from pprint import pprint
from ipdb import set_trace as brpt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torch.linalg import norm as tnorm
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.tuner.tuning import Tuner

# Global
DATA_ROOT = Path("/data/hayley-old/maptiles_v2/")

# Select Visible GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from src.data.datasets.maptiles import MaptilesDataset, MapStyles
from src.data.datamodules.maptiles_datamodule import MaptilesDataModule

from src.data.transforms.transforms import Identity
from src.models.plmodules.three_fcs import ThreeFCs

from src.visualize.utils import show_timgs
from collections import OrderedDict

from src.models.plmodules.vanilla_vae import VanillaVAE
from src.data.datamodules.maptiles_datamodule import MaptilesDataModule
from src.data.datamodules.mnist_datamodule import MNISTDataModule

# Instantiate data module
cities = ['paris']
# styles = ['OSMDefault', 'CartoVoyagerNoLabels', 'StamenTonerBackground']
styles = ['CartoVoyagerNoLabels', 'StamenTonerBackground']

zooms = ['15']
n_channels = 3
dset = MaptilesDataset(
    data_root=DATA_ROOT,
    cities=cities,
    styles=styles,
    zooms=zooms,
    n_channels=n_channels
)

in_shape = (n_channels, 64, 64)
dm = MaptilesDataModule.from_maptiles_dataset(dset, in_shape=in_shape)
dm.setup('fit')
print(dm.train_ds.channel_mean, dm.train_ds.channel_std)
print(dm.train_ds.transform)
print(dm.val_ds.transform)
# print(dm.test_ds.transform)


# Instantiate the pl Module
latent_dim = 10
hidden_dims = [32,64,128,256]#,512]
act_fn = nn.ReLU()
lr = 1e-3
model = VanillaVAE(in_shape=dm.size(), #dm.in_shape,
                    latent_dim=latent_dim,
                    hidden_dims=hidden_dims,
                    act_fn=act_fn,
                  learning_rate=lr)
# print(dm.hparams)
print(model.hparams)


# Instantiate a PL `Trainer` object
# -- most basic trainer: uses good defaults, eg: auto-tensorboard logging, checkpoints, logs, etc.
# -- Pass the data module along with a pl module
# ref: https://www.learnopencv.com/tensorboard-with-pytorch-lightning/
# Callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

tb_logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs', name='vae_maptiles')
trainer_config = {
    'gpus':1,
    'max_epochs': 100,
    'progress_bar_refresh_rate':20,
#     'auto_lr_find': True,
    'terminate_on_nan':True,
#     'num_sanity_val_steps':0.25,
    'check_val_every_n_epoch':10,
    'logger':tb_logger,
#     'callbacks':[EarlyStopping('val_loss')]
}
trainer = pl.Trainer(**trainer_config)
# trainer.tune(model)
# Fit model
trainer.fit(model, dm)


