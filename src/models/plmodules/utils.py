from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load

def get_best_ckpt(model: pl.LightningModule,
                  verbose:bool = False):
    ckpt_path = model.trainer.checkpoint_callback.best_model_path

    if verbose:
        ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)  # dict object
        for k, v in ckpt.items():
            if 'state' in k:
                continue
            if isinstance(v, dict):
                pprint(f"{k}")
                pprint(f"{v}")
            else:
                pprint(f"{k}:{v}")
    return ckpt_path

def load_model(model: pl.LightningModule, ckpt_path: str):
    # Inplace loading of the model state from the ckpt_path
    # ckpt_path = get_best_ckpt((model))
    ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)  # dict object
    model.load_state_dict(ckpt['state_dict'])

def load_best_model(model: pl.LightningModule):
    """
    Load the model state from the best ckpt_path recorded during the training
    Update the model's state **inplace**

    :param model: pl.LightningModule
    :return:
    """
    ckpt_path = get_best_ckpt((model))
    ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)  # dict object
    model.load_state_dict(ckpt['state_dict'])