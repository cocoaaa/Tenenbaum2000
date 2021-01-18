"""
train_bivae.py

Required args:
    --model_name: "vae" or "iwae"
    --data_name: "maptiles" or "mnist"
    --latent_dim: int, eg. 10

Optional args: (partial)
    --hidden_dims: eg. --hidden_dims 32 64 128 256 (which is default)

To run: (at the root of the project, ie. /data/hayley-old/Tenanbaum2000
nohup python train_bivae.py --model_name="vae" --data_name="mnist" --latent_dim=10
nohup python train_bivae.py --model_name="iwae" --data_name="mnist" --latent_dim=10
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64  --adv_dim 32 32 -lr 1e-3 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1

"""
import os,sys
import re
import math
from datetime import datetime
import time
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
import warnings
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers

# plmodules
from src.models.plmodules.three_fcs import ThreeFCs
from src.models.plmodules.vanilla_vae import VanillaVAE
from src.models.plmodules.iwae import IWAE
from src.models.plmodules.bilatent_vae import BiVAE

# datamodules
from src.data.datamodules.maptiles_datamodule import MaptilesDataModule
from src.data.datamodules.mnist_datamodule import MNISTDataModule
from src.data.datamodules import MultiMonoMNISTDataModule

# callbacks
from src.callbacks.recon_logger import ReconLogger
from src.callbacks.hist_logger import  HistogramLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from src.callbacks.beta_scheduler import BetaScheduler


# src helpers
from src.utils.misc import info
from src.models.model_wrapper import ModelWrapper
from src.utils.misc import info, n_iter_per_epoch


def get_act_fn(fn_name:str) -> Callable:
    fn_name = fn_name.lower()
    return {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
    }[fn_name]


def get_dm_class(dm_name:str) -> object:
    dm_name = dm_name.lower()
    return {
        'mnist': MNISTDataModule,
        'maptiles': MaptilesDataModule,
        'multi_mono_mnist': MultiMonoMNISTDataModule,
    }[dm_name]


def get_model_class(model_name: str) -> object:
    model_name = model_name.lower()
    return {
        "three_fcs": ThreeFCs,
        "vae": VanillaVAE,
        "iwae": IWAE,
        "bivae": BiVAE,

    }[model_name]

def instantiate_dm(args):
    data_name = args.data_name.lower()
    data_root = Path(args.data_root)

    if data_name == 'mnist':
        kwargs = {
            'data_root': data_root,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'verbose': args.verbose,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
        }
        dm = MNISTDataModule(**kwargs)

    elif data_name == 'maptiles':
        kwargs = {
            'data_root': data_root,
            'cities': args.cities,
            'styles': args.styles,
            'zooms': args.zooms,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'verbose': args.verbose,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
        }
        dm = MaptilesDataModule(**kwargs)
    elif data_name == 'multi_mono_mnist':
        kwargs = {
            'data_root': args.data_root,
            'colors': args.colors,
            'seed': args.seed,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'verbose': args.verbose,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
        }
        dm = MultiMonoMNISTDataModule(**kwargs)
    else:
        raise KeyError("Data name must be in ['mnist', 'maptiles']")

    return dm



def instantiate_model(args):
    act_fn = get_act_fn(args.act_fn)

    # Base init kwargs
    kwargs = {
        'in_shape': args.in_shape, #dm.size()
        'latent_dim': args.latent_dim,
        'hidden_dims': args.hidden_dims,
        'act_fn': act_fn,
        'learning_rate': args.learning_rate,
        'verbose': args.verbose,
    }
    model_name = args.model_name
    model_class = get_model_class(model_name)

    # Specify extra kwargs for each model class
    # Add one for new model here
    if model_name == 'iwae':
        kwargs['n_samples'] = args.n_samples

    if model_name == 'bivae':
        extra_kw = {
            "n_styles": args.n_styles,
            "adversary_dims": args.adversary_dims,
            "is_contrasive": args.is_contrasive,
            "kld_weight": args.kld_weight,
            "adv_loss_weight": args.adv_loss_weight,
        }
        kwargs.update(extra_kw)

    return model_class(**kwargs)


if __name__ == '__main__':

    parser = ArgumentParser()

    # Define general arguments for this CLI script for trianing/testing
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--mode", type=str, default='fit', help="fit or test")
    parser.add_argument("--log_root", type=str, default='./lightning_logs', help='root directory to save lightning logs')
    parser.add_argument("--gpu_id", type=str, required=True, help="ID of GPU to use")

    # Callback args
    # parser.add_argument("--hist_epoch_interval", type=int, default=10, help="Epoch interval to plot histogram of q's parameter")
    # parser.add_argument("--recon_epoch_interval", type=int, default=10, help="Epoch interval to plot reconstructions of train and val samples")

    parser.add_argument("-v", "--verbose", action="store_true", default=False)

    args, unknown = parser.parse_known_args()
    print("CLI args: ")
    pprint(args)

    # ------------------------------------------------------------------------
    # Add model/datamodule/trainer specific args
    # ------------------------------------------------------------------------
    model_class = get_model_class(args.model_name)
    dm_class = get_dm_class(args.data_name)
    parser = model_class.add_model_specific_args(parser)
    parser = dm_class.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    # Callback switch args
    parser = BetaScheduler.add_argparse_args(parser)

    args = parser.parse_args()
    print("Final args: ")
    pprint(args)
    # ------------------------------------------------------------------------
    # Initialize model, datamodule, trainer using the parsered arg dict
    # ------------------------------------------------------------------------
    # Select Visible GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Init. datamodule and model
    dm = instantiate_dm(args)
    dm.setup('fit')
    model = instantiate_model(args)

    # Specify logger
    exp_name = f'{model.name}_{dm.name}'
    print('Exp name: ', exp_name)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_root,
                                             name=exp_name,
                                             default_hp_metric=False,
                                             )
    log_dir = Path(tb_logger.log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
        print("Created: ", log_dir)

    # Specify callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='epoch')
        # HistogramLogger(hist_epoch_interval=args.hist_epoch_interval),
        # ReconLogger(recon_epoch_interval=args.recon_epoch_interval),
        #         EarlyStopping('val_loss', patience=10),
    ]
    if args.use_beta_scheduler:
        # n_epoch
        max_iters = n_iter_per_epoch(dm.train_dataloader()) * args.max_epochs
        callbacks.append(BetaScheduler(max_iters,
                                       start=args.beta_start,
                                       stop=args.beta_stop,
                                       n_cycle=args.beta_n_cycle,
                                       ratio=args.beta_ratio,
                                       log_tag=args.beta_log_tag))

    overwrites = {
        'gpus':1, #use a single gpu
         'progress_bar_refresh_rate':0, # don't print out progress bar
        'terminate_on_nan':True,
        'check_val_every_n_epoch':10,
        'logger': tb_logger,
        'callbacks': callbacks
    }

    # Init. trainer
    trainer = pl.Trainer.from_argparse_args(args, **overwrites)

    # Log model's computational graph
    model_wrapper = ModelWrapper(model)
    # tb_logger.experiment.add_graph(model_wrapper, model.)
    tb_logger.log_graph(model_wrapper)

    # ------------------------------------------------------------------------
    # Run the experiment
    # ------------------------------------------------------------------------
    start_time = time.time()
    print(f"{exp_name} started... Logging to {Path(tb_logger.log_dir).absolute()}")
    trainer.fit(model, dm)
    print(f"Finished at ep {trainer.current_epoch, trainer.batch_idx}")


    # ------------------------------------------------------------------------
    # Log the best score and current experiment's hyperparameters
    # ------------------------------------------------------------------------
    hparams = model.hparams.copy()
    hparams.update(dm.hparams)
    best_score = trainer.checkpoint_callback.best_model_score.item()
    metrics = {'hparam/best_score': best_score}  # todo: define a metric and use it here
    trainer.logger.log_hyperparams(hparams, metrics)

    print("Logged hparams and metrics...")
    print("\t hparams: ")
    pprint(hparams)
    print("=====")
    print("\t metrics: ", metrics)

    # ------------------------------------------------------------------------
    # Evaluation
    #   1. Reconstructions:
    #     x --> model.encoder(x) --> theta_z
    #     --> sample N latent codes from the Pr(z; theta_z)
    #     --> model.decoder(z) for each sampled z's
    #   2. Embedding:
    #       a mini-batch input -> mu_z, logvar_z
    #       -> rsample
    #       -> project to 2D -> visualize
    #   3. Inspect the topology/landscape of the learned latent space
    #     Latent traversal: Pick a dimension of the latent space.
    #     - Keep all other dimensions' values constant.
    #     - Vary the chosen dimenion's values (eg. linearly, spherically)
    #     - and decode the latent codes. Show the outputs of the decoder.
    #   4. Marginal Loglikelihood of train/val/test dataset
    # ------------------------------------------------------------------------
    print("Evaluations...")
    model.eval()

    # ------------------------------------------------------------------------
    # 1. Recon
    # ------------------------------------------------------------------------


    print(f"Done: took {time.time() - start_time}")




