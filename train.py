"""
trainer_main.py

Required args:
    --model_name: "vae" or "iwae"
    --data_name: "maptiles" or "mnist"
    --latent_dim: int, eg. 10

Optional args: (partial)
    --hidden_dims: eg. --hidden_dims 32 64 128 256 (which is default)

To run: (at the root of the project, ie. /data/hayley-old/Tenanbaum2000
nohup python src/train.py --model_name="vae" --data_name="mnist" --latent_dim=10
nohup python src/train.py --model_name="iwae" --data_name="mnist" --latent_dim=10

Jan 23, 2021
nohup python train.py --model_name="vae" --data_name="maptiles" --latent_dim=10 \
--cities berlin rome \
--styles StamenTonerBackground \
--zooms 14 \
--in_shape 1 64 64 \
--batch_size 32 \
--hidden_dims 32 32 64 64 128 \
--gpu_id 2 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23/" &



nohup python train.py --model_name="vae" --data_name="maptiles" --latent_dim=10 \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' 'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles StamenTonerBackground \
--zooms 14 \
--in_shape 1 64 64 \
--batch_size 32 \
--hidden_dims 32 32 64 64 128 \
--gpu_id 2 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23/" &


nohup python train.py --model_name="vae" --data_name="maptiles" --latent_dim=10 \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' 'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles StamenTonerBackground \
--zooms 14 \
--in_shape 1 64 64 \
--batch_size 32 \
--hidden_dims 32 64 64 64 128 \
--gpu_id 2 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23/" &


nohup python train.py --model_name="vae" --data_name="maptiles" --latent_dim=10 \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' 'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles StamenTonerBackground \
--zooms 14 \
--in_shape 1 64 64 \
--batch_size 32 \
--hidden_dims 32 64 64 128 256 \
--gpu_id 1 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23/" &

nohup python train.py --model_name="vae" --data_name="maptiles" --latent_dim=20 \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' 'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles StamenTonerBackground \
--zooms 14 \
--in_shape 1 64 64 \
--batch_size 32 \
--hidden_dims 32 64 64 128 256 \
--gpu_id 1 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23/" &


"""
# Load libraries
# In[2]:


import os,sys
from datetime import datetime
import time
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

# callbacks
from src.callbacks.recon_logger import ReconLogger
from src.callbacks.hist_logger import  HistogramLogger

# src helpers
from src.utils.misc import info
from src.models.model_wrapper import ModelWrapper

# utils for instatiating a selected datamodule and a selected model
from .utils import get_model_class, get_dm_class
from .utils import instantiate_model, instantiate_dm


if __name__ == '__main__':

    parser = ArgumentParser()

    # Define general arguments for this CLI script for trianing/testing
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--mode", type=str, default='fit', help="fit or test")
    parser.add_argument("--log_root", type=str, default='./lightning_logs', help='root directory to save lightning logs')
    parser.add_argument("--gpu_id", type=str, required=True, help="ID of GPU to use")

    # Callback args
    parser.add_argument("--hist_epoch_interval", type=int, default=10, help="Epoch interval to plot histogram of q's parameter")
    parser.add_argument("--recon_epoch_interval", type=int, default=10, help="Epoch interval to plot reconstructions of train and val samples")

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

    # Specify logger and callbacks
    exp_name = f'{model.name}_{dm.name}'
    print('Exp name: ', exp_name)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_root,
                                             name=exp_name,
                                             default_hp_metric=False,
                                             )
    callbacks = [
        HistogramLogger(hist_epoch_interval=args.hist_epoch_interval),
        ReconLogger(recon_epoch_interval=args.recon_epoch_interval),
        #         EarlyStopping('val_loss', patience=10),
    ]

    overwrites = {
        'gpus':1,
         'progress_bar_refresh_rate':0,
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




