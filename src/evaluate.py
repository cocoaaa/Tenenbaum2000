import torch
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
from pprint import pprint
import torchvision
from src.data.transforms.transforms import Identity, Unnormalizer, LinearRescaler
from src.data.transforms.functional import unnormalize
from src.visualize.utils import show_timgs, show_batch
from src.utils.misc import info

def show_recon(model,
               tb_logger=None,
               unnorm: bool = True,
               to_show: bool = True, verbose: bool = False):
    model.eval()
    dm = model.trainer.datamodule
    cmap = 'gray' if dm.size()[0] == 1 else None
    train_mean, train_std = dm.train_mean, dm.train_std
    with torch.no_grad():
        for mode in ['train', 'val']:
            dl = getattr(model, f"{mode}_dataloader")()
            x, y = next(iter(dl))
            x = x.to(model.device)
            x_recon = model.generate(x)

            x = x.cpu()
            x_recon = x_recon.cpu()

            if verbose:
                info(x, f"{mode}_x")
                info(x_recon, f"{mode}_x_recon")

            if unnorm:
                x_unnormed = unnormalize(x, train_mean, train_std)
                x_recon_unnormed = unnormalize(x_recon, train_mean, train_std)
                if verbose:
                    print("===After unnormalize===")
                    info(x_unnormed, f"{mode}_x_unnormed")
                    info(x_recon_unnormed, f"{mode}_x_recon_unnormed")

            if to_show:
                _x = x_unnormed if unnorm else x
                _x_recon = x_recon_unnormed if unnorm else x_recon
                show_timgs(_x, title=f"Input: {mode}", cmap=cmap)
                #                 show_timgs(_x_recon, title=f"Recon: {mode}", cmap=cmap)
                show_timgs(LinearRescaler()(_x_recon), title=f"Recon(linearized): {mode}", cmap=cmap)

            # Log input-recon grid to TB
            if tb_logger is not None:
                input_grid = torchvision.utils.make_grid(x_unnormed)  # (C, gridh, gridw)
                recon_grid = torchvision.utils.make_grid(x_recon_unnormed)  # (C, gridh, gridw)
                #         normed_recon_grid = torchvision.utils.make_grid(LinearRescaler()(x_recon_unnormed))
                grid = torch.cat([input_grid, recon_grid], dim=-1)  # inputs | recons
                tb_logger.experiment.add_image(f"{mode}/recons", grid, global_step=0)
