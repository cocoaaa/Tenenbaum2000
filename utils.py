from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar

import torch
import torch.nn as nn

# plmodules
from src.models.plmodules.three_fcs import ThreeFCs
from src.models.plmodules.vanilla_vae import VanillaVAE
from src.models.plmodules.iwae import IWAE
from src.models.plmodules.bilatent_vae import BiVAE

# datamodules
from src.data.datamodules.maptiles_datamodule import MaptilesDataModule
from src.data.datamodules.mnist_datamodule import MNISTDataModule
from src.data.datamodules import MultiMonoMNISTDataModule
from src.data.datamodules import MultiRotatedMNISTDataModule
from src.data.datamodules import MultiMaptilesDataModule

# src helpers
from src.utils.misc import info

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
        'multi_rotated_mnist': MultiRotatedMNISTDataModule,
        'multi_maptiles': MultiMaptilesDataModule,
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
            'data_root': data_root,
            'colors': args.colors,
            'seed': args.seed,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
            'verbose': args.verbose,
        }
        dm = MultiMonoMNISTDataModule(**kwargs)

    elif data_name == 'multi_rotated_mnist':
        kwargs = {
            'data_root': data_root,
            'angles': args.angles,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
            'verbose': args.verbose,
            'split_seed': args.split_seed,
        }
        dm = MultiRotatedMNISTDataModule(**kwargs)

    elif data_name == 'multi_maptiles':
        kwargs = {
            'data_root': data_root,
            'cities': args.cities,
            'styles': args.styles,
            'zooms': args.zooms,
            'in_shape': args.in_shape,
            'batch_size': args.batch_size,
            'pin_memory': args.pin_memory,
            'num_workers': args.num_workers,
            'verbose': args.verbose,
        }
        dm = MultiMaptilesDataModule(**kwargs)

    # TODO: Add new data modules here

    else:
        raise KeyError(
            "Data name must be in ['mnist', 'maptiles', 'multi_mono_mnist', 'multi_rotated_mnist', 'multi_maptiles']"
        )

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