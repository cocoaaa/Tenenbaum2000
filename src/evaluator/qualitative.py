from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torchvision

import pytorch_lightning as pl

from src.data.transforms.transforms import LinearRescaler
from src.data.transforms.functional import unnormalize

from src.models.plmodules import BiVAE
from src.visualize.utils import show_timg, show_timgs, show_batch
from src.utils.misc import info, get_next_version_path


def show_recon(model: pl.LightningModule,
               dm: pl.LightningDataModule,
               tb_writer: SummaryWriter = None,
               global_step: int = 0,
               unnorm: bool = True,
               to_show: bool = True,
               verbose: bool = False):
    model.eval()
    cmap = 'gray' if dm.size()[0] == 1 else None
    train_mean, train_std = dm.train_mean, dm.train_std
    with torch.no_grad():
        for mode in ['train', 'val']:
            dl = getattr(dm, f"{mode}_dataloader")()
            batch = next(iter(dl))

            if isinstance(batch, dict):
                x = batch['img']
                #             label_c = batch['digit']  # digit/content label (int) -- currently not used
                #             label_s = batch['color']
            else:
                x, _ = batch

            x = x.to(model.device)
            x_recon = model.generate(x)

            # Move to cpu for visualization
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
            if tb_writer is not None:
                input_grid = torchvision.utils.make_grid(x_unnormed)  # (C, gridh, gridw)
                recon_grid = torchvision.utils.make_grid(x_recon_unnormed)  # (C, gridh, gridw)
                normed_recon_grid = torchvision.utils.make_grid(LinearRescaler()(x_recon_unnormed))

                grid = torch.cat([input_grid, normed_recon_grid], dim=-1)  # inputs | recons
                tb_writer.add_image(f"{mode}/recons", grid, global_step=global_step)


# ------------------------------------------------------------------------
# Latent Space
# ------------------------------------------------------------------------
def get_class_reps(dl: DataLoader) -> Dict[Union[str,int], torch.Tensor]:
    class_reps = {}
    for i in range(len(dl.dataset)):
        batch = dl.dataset[i]
        x = batch['img']
        label_c = batch['digit']  # digit/content label (int) -- currently not used
        label_s = batch['color']

        if len(class_reps) >= 10:
            break
        if isinstance(label_c, torch.Tensor):
            label_c = label_c.item()
        label_c = str(label_c)
        if label_c in class_reps:
            continue
        class_reps[label_c] = x
    return class_reps

def evaluate_transfers(model: BiVAE,
                    constant_code: str, #"c" or "s"
                    class_reps: Dict[str, torch.Tensor],
                    log_dir: Path,
                    train_mean: List, # to unnormalize output tensor
                    train_std: List, # to unnormalize
                    linearlize: bool=True):
    """

    :param model: Trained BiVAE model
    :param class_reps: a dictionary of string class_id <-> a single 3dim Tensor (C,H,W)
    :param log_dir: Path to the model.logger's log_dir (Eg. '..../exp_name/version7')
    :param train_mean: Original datamodule's training set's mean
    :param train_std:  Oiriginal datamodule's training set std
    :param linearlize: (bool). If true, linearlize the output image to range [0,1] for better viz. contrast
    :return:
    """
    assert constant_code in ['c', 's'], "constant_code must be 'c' or 's' (for content, style, respectively"
    model.eval()
    ids = sorted(class_reps.keys())
    grids = {}
    for i, id_a in enumerate(ids):

        grids[id_a] = []
        for j, id_b in enumerate(ids):

            img_a = class_reps[id_a]
            img_b = class_reps[id_b]
            img_pair = torch.stack([img_a, img_b], dim=0)
            unnormed_img_pair = unnormalize(img_pair, train_mean, train_std)

            with torch.no_grad():
                dict_qparams = model.encode(img_pair)
                dict_z = model.rsample(dict_qparams)

                if constant_code == 'c':
                    # Fix content to c[0]
                    content = dict_z["c"][[0]]
                    style = dict_z["s"][[1]]
                elif constant_code == 's':
                    # Fix style to s[0]
                    content = dict_z["c"][[1]]
                    style = dict_z["s"][[0]]
                test_dict_z = {"c": content, "s": style}

                # Reconstruct
                z = model.combine_content_style(test_dict_z)
                recons = model.decode(z)

                # Optional: for better viz, unnormalize or/and linearlize
                unnormed_recons = unnormalize(recons, train_mean, train_std)
                if linearlize:
                    img_pair = LinearRescaler()(img_pair) #todo
                    unnormed_recons = LinearRescaler()(unnormed_recons)

                grid = torchvision.utils.make_grid(
                    torch.cat([unnormed_img_pair, unnormed_recons], dim=0)
                )  # (3, gridh, gridw)
                grids[id_a].append(grid)


    # Save each content's transferred image as:
    # log_dir/content_transfers/{next_version}/"content_transfers_{content-id}.png"
    constant_type = "content" if constant_code == "c" else "style"
    save_dir = get_next_version_path(log_dir, name=f'{constant_type}_transfers')

    if not save_dir.exists():
        save_dir.mkdir()
        print("Created: ", save_dir)

    for id_a, recons in grids.items():
        recons = torch.cat(recons, dim=1)
        save_path = save_dir/f'{constant_type}_transfers_{id_a}.png'
        show_timg(recons,
                  title=id_a,
                  save_path=save_path,
                 )



def save_content_transfers(model: BiVAE, *args, **kwargs):
    """

    :param model: Trained BiVAE model
    :param class_reps: a dictionary of string class_id <-> a single 3dim Tensor (C,H,W)
    :param log_dir: Path to the model.logger's log_dir (Eg. '..../exp_name/version7')
    :param train_mean: Original datamodule's training set's mean
    :param train_std:  Oiriginal datamodule's training set std
    :param linearlize: (bool). If true, linearlize the output image to range [0,1] for better viz. contrast
    :return:
    """
    evaluate_transfers(model, constant_code='c', *args, **kwargs)

def save_style_transfers(model: BiVAE, *args, **kwargs):
    """

    :param model: Trained BiVAE model
    :param class_reps: a dictionary of string class_id <-> a single 3dim Tensor (C,H,W)
    :param log_dir: Path to the model.logger's log_dir (Eg. '..../exp_name/version7')
    :param train_mean: Original datamodule's training set's mean
    :param train_std:  Oiriginal datamodule's training set std
    :param linearlize: (bool). If true, linearlize the output image to range [0,1] for better viz. contrast
    :return:
    """
    evaluate_transfers(model, constant_code='s', *args, **kwargs)

def run_both_transfers(model: BiVAE, *args, **kwargs):
    """
    Run both content-transfer and style-transfer on the each pair of the content-representative tensor images
    :param model: Trained BiVAE model
    :param class_reps: a dictionary of string class_id <-> a single 3dim Tensor (C,H,W)
    :param log_dir: Path to the model.logger's log_dir (Eg. '..../exp_name/version7')
    :param train_mean: Original datamodule's training set's mean
    :param train_std:  Oiriginal datamodule's training set std
    :param linearlize: (bool). If true, linearlize the output image to range [0,1] for better viz. contrast
    :return:
    """
    save_content_transfers(model, *args, **kwargs)
    save_style_transfers(model, *args, **kwargs)





















