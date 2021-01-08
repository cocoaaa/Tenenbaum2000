#!/usr/bin/env python
# coding: utf-8

# # Train Adversarial VAE (BiVAE) on 3 styles of Monochrome MNIST datasets
# - Jan 6, 2021
# 

# ## Load libraries

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os,sys
import re
import math
from datetime import datetime
import time
sys.dont_write_bytecode = True


# In[3]:


import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar

from pprint import pprint
from ipdb import set_trace as brpt


# In[4]:


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torch.linalg import norm as tnorm
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.tuner.tuning import Tuner


# Select Visible GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# ## Set Path 
# 1. Add project root and src folders to `sys.path`
# 2. Set DATA_ROOT to `maptile_v2` folder

# In[5]:


this_nb_path = Path(os.getcwd())
ROOT = this_nb_path.parent
SRC = ROOT/'src'
DATA_ROOT = Path("/data/hayley-old/maptiles_v2/")
paths2add = [this_nb_path, ROOT]

print("Project root: ", str(ROOT))
print('Src folder: ', str(SRC))
print("This nb path: ", str(this_nb_path))


for p in paths2add:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
        print(f"\n{str(p)} added to the path.")
        
# print(sys.path)


# In[6]:


from src.data.transforms.transforms import Identity, Unnormalizer, LinearRescaler
from src.data.transforms.functional import unnormalize

from src.visualize.utils import show_timg, show_timgs, show_batch, make_grid_from_tensors
from src.utils.misc import info, get_next_version_path
from collections import OrderedDict


# ## Start experiment 
# Given a maptile, predict its style as one of OSM, CartoVoyager

# In[7]:


from src.models.plmodules.vanilla_vae import VanillaVAE
from src.models.plmodules.bilatent_vae import BiVAE
from src.models.plmodules.three_fcs import ThreeFCs


# In[ ]:


# # For reproducibility, set seed like following:
# seed = 100
# pl.seed_everything(seed)
# # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
# model = Model()
# trainer = pl.Trainer(deterministic=True)


# ## Adversarial model
# 
# TODO:
# 
# ---
# ### For batch in dataloader:
# - x: (BS, C, h, w): a mini-batch of (c,h,w) tensor
# 
# ### mu, log_var = model.encoder(x) 
# - mu: (BS, latent_dim)
# - log_var: (BS, latent_dim)
# 
# ### z = model.rsample(mu, log_var, self.n_samples) 
# - z: (BS, n_samples, latent_dim)
# -`z[n]` constains `n_samples` number of latent codes, sampled from the same distribution `N(mu[n], logvar[n])`
#  
# ### recon = model.decoder(z) 
# - recon: (BS, n_samples, c, h, w)
# - `recon[n]` contains `n_samples` number of (c,h,w)-sized $mu_{x}$, corresponding to the center of the factorized Gaussian for the latent code $z^{(n,l)}$ ($l$th z_sample from $N(\mu[n], logvar[n])$, ie. $\mu_{x}^{(n,l)}$
# 
# ### out = model.forward(x)
# - out (dict): keys are "mu", "logvar", "recon"
# 
# ### loss_dict = loss_function(out, x, self.n_samples)
# - loss_dict (dict): keys are "loss", "kl", "recon_loss"
# - kl is computed the same way as in the Vanillia_VAE model's `loss_function`
# - recon_loss is a generalized version with `self.n_samples` (>=1) number of samples to estimated each datapoint's MSE_loss as the average over the loss's from the `n_samples` number of $z_{n,l}$ samples.
# 

# In[8]:


from src.data.datasets import MultiMonoMNIST
from src.data.datamodules import MultiMonoMNISTDataModule
from src.models.plmodules.bilatent_vae import BiVAE


# In[10]:


# Dataset settings
data_dir = Path("/data/hayley-old/Tenanbaum2000/data/Mono-MNIST/")
colors = ['red', 'green', 'blue']
seed = 123
in_shape = (3, 32,32)
batch_size = 128


# In[11]:


# Init MNIST-M DataModule
# dm = MNISTMDataModule(
#     data_root=data_root, 
#     in_shape=in_shape,
#     batch_size=batch_size)
# dm.setup('fit')
# # show_batch(dm, cmap='gray')


# In[12]:


# Create a multisource mono-mnist datamodule
dm = MultiMonoMNISTDataModule(
    data_root=data_dir,
    colors=colors,
    seed=seed,
    in_shape=in_shape,
    batch_size=batch_size,
    shuffle=True,
)
dm.setup('fit')


# In[ ]:


print(dm.name)
dl = dm.train_dataloader()
# Show a batch
x,y = next(iter(dl))
show_timgs(x)
info(x)
print(y)
print("===")


# In[14]:


# Initi plModule
latent_dim=10
hidden_dims = [32,64,128,256] #,512]
adversary_dims = [30,20,15]
lr = 1e-3
act_fn = nn.ReLU()
is_contrasive = True # If true, use adv. loss from both content and style codes. Else just style codes
model = BiVAE(
    in_shape=dm.size(), 
    n_classes=dm.n_classes,
    latent_dim=latent_dim,
    hidden_dims=hidden_dims,
    adversary_dims=adversary_dims,
    learning_rate=lr, 
    act_fn=act_fn,
    size_average=False
)

# model
    


# In[ ]:


# Add Callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.callbacks.hist_logger import HistogramLogger
from src.callbacks.recon_logger import ReconLogger

# Model wrapper from graph viz
from src.models.model_wrapper import ModelWrapper

callbacks = [
#         HistogramLogger(hist_epoch_interval=1),
#         ReconLogger(recon_epoch_interval=1),
#         EarlyStopping('val_loss', patience=10),
]

# Start the experiment
exp_name = f'{model.name}_{dm.name}'
tb_logger = pl_loggers.TensorBoardLogger(save_dir=f'{ROOT}/temp-logs', 
                                         name=exp_name,
                                         log_graph=False,
                                        default_hp_metric=False)
print(tb_logger.log_dir)

# Log computational graph
# model_wrapper = ModelWrapper(model)
# tb_logger.experiment.add_graph(model_wrapper, model.example_input_array.to(model.device))
# tb_logger.log_graph(model)

trainer_config = {
    'gpus':1,
    'max_epochs': 200,
    'progress_bar_refresh_rate':20,
#     'auto_lr_find': True,
    'terminate_on_nan':True,
#     'num_sanity_val_steps':0.25,
    'check_val_every_n_epoch':10,
    'logger':tb_logger,
#     'callbacks':callbacks,
}

# 
# trainer = pl.Trainer(fast_dev_run=3)
trainer = pl.Trainer(**trainer_config)
# trainer.tune(model=model, datamodule=dm)

# Start exp
# Fit model
trainer.fit(model, dm)
print(f"Finished at ep {trainer.current_epoch, trainer.batch_idx}")


# In[ ]:





# ## Log  hparmeters and `best_score` to tensorboard

# In[20]:


hparams = model.hparams.copy()
hparams.update(dm.hparams)
best_score = trainer.checkpoint_callback.best_model_score.item()
metrics = {'hparam/best_score': best_score} #todo: define a metric and use it here
pprint(hparams)
pprint(metrics)


# In[21]:


# Use pl.Logger's method "log_hyperparameters" which handles the 
# hparams' element's formats to be suitable for Tensorboard logging
# See: 
# https://sourcegraph.com/github.com/PyTorchLightning/pytorch-lightning@be3e8701cebfc59bec97d0c7717bb5e52afc665e/-/blob/pytorch_lightning/loggers/tensorboard.py#explorer:~:text=def%20log_hyperparams
best_score = trainer.checkpoint_callback.best_model_score.item()
metrics = {'hparam/best_score': best_score} #todo: define a metric and use it here
trainer.logger.log_hyperparams(hparams, metrics)


# In[ ]:





#  TODO:
#  OPTIMIZER
#  def configure_optimizers(self):
#         #TODO: ADD optimizer for discriminator
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.get("learning_rate"))

# ## TODO: 
# Showing the changes in the scores based on c and scores based on s will be super intersting to see as the model learns!!!

# ## Evaluations

# In[22]:


from src.models.plmodules.utils import get_best_ckpt, load_model, load_best_model
from pytorch_lightning.utilities.cloud_io import load as pl_load


# Load best model recorded during the training
# 

# In[32]:


ckpt_path = get_best_ckpt(model,verbose=True)
ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage) #dict
best_epoch = ckpt['epoch']
best_global_step = ckpt['global_step']

# Load model state from the best ckpt
model.load_state_dict(ckpt['state_dict'])


# ### Reconstruction
#     
#     

# In[24]:


def show_recon(model, 
               tb_logger=None,
               global_step:int=0,
               unnorm:bool=True, 
               to_show:bool=True, 
               verbose:bool=False):
    model.eval()
    dm = model.trainer.datamodule
    cmap = 'gray' if dm.size()[0] ==1 else None
    train_mean, train_std = dm.train_mean, dm.train_std
    with torch.no_grad():
        for mode in ['train', 'val']:
            dl = getattr(model, f"{mode}_dataloader")()
            x,y = next(iter(dl))
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
                input_grid = torchvision.utils.make_grid(x_unnormed) # (C, gridh, gridw)
                recon_grid = torchvision.utils.make_grid(x_recon_unnormed) # (C, gridh, gridw)
                normed_recon_grid = torchvision.utils.make_grid(LinearRescaler()(x_recon_unnormed))
                
                grid = torch.cat([input_grid, normed_recon_grid], dim=-1) #inputs | recons
                tb_logger.experiment.add_image(f"{mode}/recons", grid, global_step=global_step)


# In[33]:


show_recon(model, tb_logger, global_step=best_global_step, verbose=True)


# ## Separation of content vs. style in the latent space

# - Case 1: Take two inputs of same content (digit id) and very different colorization styles
# - Case 2: Take two inputs of different contents (eg. digit 0 and digit 5)
#     - First, pass the first image to encoder -> get c^(1) and s^(1)
#     - Now pass the second image to encoder -> get c^(2) and s^(2)
#     - recon1 = decoder(z=[c^(1), s^(1)])
#     - recon2 = decoder(z=[c^(2), s^(1)])
#     - ---
#     - recon3 = decoder(z=[c^(1), s^(2)])
#     - recon4 = decoder(z=[c^(2), s^(2)])
# ---
# Do it for a dataset from multiple styles
# - Create a concatenated dataseta

# ### Collect one image per digit id to create a fixed evaluation image set
# Save the test image set to compare different models 

# In[43]:


import warnings
def get_class_reps(dl: DataLoader) -> Dict[Union[str,int], torch.Tensor]:
    class_reps = {}
    for i in range(len(dl.dataset)):
        x,y = dl.dataset[i]
        if len(class_reps) >= 10:
            break
        if isinstance(y, torch.Tensor):
            y = y.item()
        y = str(y)
        if y in class_reps:
            continue
        class_reps[y] = x
    return class_reps

def get_class_reps_from_dataloader(dl: DataLoader) -> Dict[Union[str,int], torch.Tensor]:
    
    class_reps = {}
    while len(class_reps) < 10:
        
        batch_x,batch_y = next(iter(dl))
        
        for x,y in zip(batch_x, batch_y):
            if len(class_reps) >= 10:
                return class_reps
            
            if isinstance(y, torch.Tensor):
                y = y.item()
            digit_id = str(y)
            class_reps[digit_id] = x
    # This should never be reached
    warnings.warn("This should never be reached. Check the source code!")
    return class_reps


# In[44]:


# Quick test
mode = 'train'
dl = getattr(dm, f"{mode}_dataloader")()


# In[50]:


class_reps = get_class_reps_from_dataloader(dl)


# In[51]:


for digit_id, timg in class_reps.items():
    show_timg(timg)
    plt.title(digit_id)


# ### Svae test data of class_reps to evaluate models trained on the datamodule,`R+G+B Multi-Mono-MNIST`. 
# These test digit-class rep. images will be used to evaluate (quantitively)
# each model's content or style transfer capabilities.

# In[61]:


from torch.utils.data import RandomSampler
from src.utils.misc import now2str


# In[72]:


seed = 1021
rng_gen = torch.Generator().manual_seed(seed)
red_mnist, green_mnist, blue_mnist = dm.train_ds.dataset.dsets
red_dl, green_dl, blue_dl = [DataLoader(ds, batch_size=32, sampler=RandomSampler(ds, generator=rng_gen))
                                        for ds in [red_mnist, green_mnist, blue_mnist]]
red_class_reps, green_class_reps, blue_class_reps = list(map(get_class_reps_from_dataloader, 
                                                             [red_dl, green_dl,blue_dl]))
class_reps_by_color = {
    'red': red_class_reps,
    'green': green_class_reps,
    'blue': blue_class_reps
}


# In[73]:


for class_reps in [red_class_reps, green_class_reps, blue_class_reps]:
    # show class-rep images
    for digit_id, timg in class_reps.items():
        show_timg(timg)
        plt.title(digit_id)


# In[74]:


# save as a pickle file
out_root = Path('/data/hayley-old/Tenanbaum2000/data/Mono-MNIST/')
out_dir = get_next_version_path(out_root, 'test_transfers')
out_dir.mkdir(parents=True, exist_ok=True)
for color, class_reps in class_reps_by_color.items():
    out_fn = out_dir/f"{color}_class_reps_seed-{seed}.pkl"
    joblib.dump(class_reps, out_fn)
    print('Saved to ', out_fn)
                


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Case 1: constant content code $c$, varying style code $s^{(i)} \in \{s^{(1)}, \dots \}$
#   - $z = [c, s^{(i)}]$ --> decoder --> $\mu^{x^{(i)}}_{pred}$ (aka. recon)
# 
# First, we fix the content code to that of a constant input (eg. an ith datapt in the training set),
# and use various style codes (eg. any $j \neq i$th datapt). We combine the (constant) content code with each of the style code and see the output of the decoder (ie. `mu_x_pred`).

# In[ ]:


# Load class reps test data as dict of tensor images
# class_reps = load_class_reps_v3()
# Reconstruct each digit-representative image by 
# interlacing a single content code with various style codes
model.eval()
train_mean, train_std = dm.train_mean, dm.train_std
ids = [str(i) for i in range(10)]
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
    #         pprint(dict_z)

            # Fix content to c[0]
            content = dict_z["c"][[0]]
            style = dict_z["s"][[1]]
            test_dict_z = {"c": content, "s": style}
    #         pprint(test_dict_z)

            # Reconstruct
            z = model.combine_content_style(test_dict_z)
            recons = model.decode(z)
            unnormed_recons = unnormalize(recons, train_mean, train_std)

            grid = torchvision.utils.make_grid(
                torch.cat([unnormed_img_pair,unnormed_recons], dim=0)
            ) # (3, gridh, gridw)
            grids[id_a].append(grid)

# Concatenate the grids to make a single grid by putting each grid in row dim(ie. dim=1)    #log_dir/content_transfers/version_x
# -- Optionally, save the image results    
log_dir = Path(model.logger.log_dir)
save_dir = get_next_version_path(log_dir, name='content_transfers') 
save_dir.mkdir()
print("Created: ", save_dir)

for id_a, recons in grids.items():
    recons = torch.cat(recons, dim=1)
    save_path = save_dir/f"content_transfers_{id_a}.png"
    show_timg(recons, 
              title=id_a, 
              save_path=save_path,
             )
    plt.axis('off')
    plt.show()
    


# ### Case 2: fix the style code, and apply the style to various contents.
# We have a single stlye code provider, and multiple content providers.
# - We first extract the style code $s$ from the style provider
# - Then, for each content provider x^{(i)}, extract the content code $c^{(i)}$
# - Combine the constant style code with each content code and pass into the decoder.
# 
# Ideal result will show that each reconstructed image preserves the content of each content provider, while stylizing the content in the style of the style provider.
# 

# In[ ]:


# Viz options
linearlize = True

# Reconstruct each digit-representative image by 
# interlacing a single content code with various style codes
model.eval()
ids = [str(i) for i in range(10)]
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
    #         pprint(dict_z)

            # Fix style to s[0]
            style = dict_z["s"][[0]]
            content = dict_z["c"][[1]]
            test_dict_z = {"c": content, "s": style}

            # Reconstruct
            z = model.combine_content_style(test_dict_z)
            recons = model.decode(z)
            
            # Optional: for better viz, unnormalize or/and linearlize
            unnormed_recons = unnormalize(recons, train_mean, train_std)
            if linearlize:
                img_pair = LinearRescaler()(img_pair)
                unnormed_recons = LinearRescaler()(unnormed_recons)
#             info(recons, 'recons')
#             info(unnormed_recons, 'unnormed_recons')    
            
            # Make a grid of [input pairs, reconstructed image of the mixed codes]
            grid = torchvision.utils.make_grid(
                torch.cat([unnormed_img_pair,unnormed_recons], dim=0)
            ) # (3, gridh, gridw)
            grids[id_a].append(grid)

# Concatenate the grids to make a single grid by putting each grid in row dim(ie. dim=1)    #log_dir/content_transfers/version_x
# -- Optionally, save the image results    
log_dir = Path(model.logger.log_dir)
save_dir = get_next_version_path(log_dir, name='style_transfers') 
save_dir.mkdir()
print("Created: ", save_dir)

for id_a, recons in grids.items():
    recons = torch.cat(recons, dim=1)
    save_path = save_dir/f"style_transfers_{id_a}.png"
    show_timg(recons, 
              title=id_a, 
              save_path=save_path,
             )
    plt.axis('off')
    plt.show()


# In[ ]:





# todo
# - [x] Get the version code from pl.log
# - [ ] Make logger for loss_c and loss_s
# - [ ] Impl. evaluation code
# 
# 
# Here
# - [ ] Impl. code for latent space traversal
# - [ ] Look at the embeddings based on z_c
#   - any clusters? -- run kmeans
#     - use digit id as labels in TB Projector
# - [ ] Look at the embeddings based on z_s
#   - any clusters? does it match clusters based on style class? 
#     - use style id as labels
#     
# ---
# Experiment adv vae model on maptiles
# - [ ] Train a model with maptiles w/ one style
# - [ ] Train a model with maptiles w/ two style (eg. tonerbackground, cartolight)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Latent space 
# - Embeddings (cluster, Nearest neighbor)

# #### Visualize embeddings
# - collect a batch of inputs -> encoder -> [mu, log_var] -> sample -> a batch of z's (embeddings)
# - use tb logger
# 

# In[ ]:


ckpt_path = trainer.checkpoint_callback.best_model_path
ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage) #dict object
# model.load_state_dict(ckpt['state_dict'])

for k,v in ckpt.items():
    if 'state' in k:
        continue
    pprint(f"{k}:{v}")


# In[ ]:


model.eval()
best_epoch = ckpt["epoch"] + 6

ds = model.train_dataloader().dataset
dl = DataLoader(ds, batch_size=20000)
with torch.no_grad():
#     x, y = next(iter(model.train_dataloader()))
    x, y = next(iter(dl))
    dict_qparams = model.encode(x)
    dict_z = model.rsample(dict_qparams)
    
#     z = out['z']
    
    # log embedding of z_c to tensorboard 
    writer = model.logger.experiment
    writer.add_embedding(dict_z['c'],
                         label_img=LinearRescaler()(x), 
                         metadata=y.tolist(),
                         global_step=best_epoch, #todo
                         tag="z_c"
                        )
    
    # log embedding of z_s to tensorboard 
    writer = model.logger.experiment
    writer.add_embedding(dict_z['s'],
                         label_img=LinearRescaler()(x), 
                         metadata=y.tolist(),
                         global_step=best_epoch, #todo
                         tag="z_s"
                        )
    
    


# In[ ]:


todo:
    project many more (Eg. the whole dataset) to embeddings
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ---
# ## Test BiVAE implementation 

# TODO:
# - [ ] Check output sizes of BiVAE's 
#     - [x] encode
#     - [x] rsample
#     - [x] combine_content_and_style
#     - [x] decode
#     - [x] forward
# - [ ] Check losses 

# In[ ]:


x,y = next(iter(dm.train_dataloader()))
info(x), info(y)


# - check `encode` and `rsample`

# In[ ]:


dict_qparams = model.encode(x)
for k,v in dict_qparams.items():
    print(f"\n{k}:  {v.shape}")
    if 'mu' in k:
        print(v[0])
    else:
        print(v[0].exp())


# In[ ]:


dict_z = model.rsample(dict_qparams)
for k,v in dict_z.items():
    print(f"\n{k}:  {v.shape}")
    print(v[0])


# - check `combine_content_style` and `decode`

# In[ ]:


z = model.combine_content_style(dict_z)
assert z.shape == (batch_size, latent_dim)
print("z shape: ", z.shape) #(BS, latent_dim)


# In[ ]:


mu_x_pred = model.decode(z)
assert mu_x_pred.shape == (batch_size, *in_shape)
print("mu_x_pred shape: ", mu_x_pred.shape)


# - Check the entire forward pass

# In[ ]:


out_dict  = model(x)
for k,v in out_dict.items():
    print(f"\n{k}:  {v.shape}")


# In[ ]:





# In[ ]:





# - Check the component's of the optimization objective (ie. loss)
#     - [x] partition_z: z -> dict_z (keys are "c" and "s")
#     - [ ] predict_y: z_partition -> scores
#     - [ ]

# In[ ]:


dict_z = model.partition_z(z)
for k,v in dict_z.items():
    print(f"{k}: {v.shape}")
    assert v.shape == (batch_size, model.content_dim)


# In[ ]:


c,s = dict_z["c"], dict_z["s"]
c.shape, s.shape


# TODO: 
# - [ ] Showing the changes in the scores based on c and scores based on s will be super intersting to see as the model learns!!!

# In[ ]:


scores_c = model.predict_y(c)
scores_s = model.predict_y(s)
assert scores_c.shape == (batch_size, model.n_classes)
assert scores_s.shape == (batch_size, model.n_classes)

print(scores_c[0]) # TODO: Showing the changes in the scores based on c and scores based on s will be super intersting to see as the model learns!!!
print(scores_s[0])


# In[ ]:


y[0]


# - check `compute_loss_c` and `compute_loss_s`
# 

# In[ ]:


loss_c = model.compute_loss_c(c)
print("loss_c: ", loss_c)


# In[ ]:


loss_s = model.compute_loss_s(s, y)
print("loss_s: ", loss_s)


# - Full loss workflow

# In[ ]:


out_dict = model(x)
loss_dict = model.loss_function(out_dict, [x,y], 'train')
pprint(loss_dict)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


a = torch.ones((5,2))
b = torch.zeros((5,3))


# In[ ]:


torch.cat([a,b], dim=1)


# In[ ]:


m = nn.LogSoftmax()
m(a).exp()


# In[ ]:





# ---
