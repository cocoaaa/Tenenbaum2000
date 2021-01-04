#!/usr/bin/env python
# coding: utf-8

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

from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.tuner.tuning import Tuner


# Select Visible GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
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
        
print(sys.path)


# In[6]:


from src.data.datasets.maptiles import MaptilesDataset, MapStyles
from src.data.datamodules.maptiles_datamodule import MaptilesDataModule

from src.data.transforms.transforms import Identity

from src.visualize.utils import show_timgs
from src.utils.misc import info
from collections import OrderedDict


# In[7]:


class MyTrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) #check if this sets train/val dataloaders
        self.num_data = len(self.train_dataloader().dataset)

    def get_dl(mode: str, dl_idx:int=0):
        if mode == 'train':
            dl = getattr(self, "train_dataloader")
        else:
            dl = getattr(self, f"{mode}_dataloaders")[dl_idx]
        print(dl)
        print(dl.dataset)
        return dl
    
    def get_next_batch(mode: str, dl_idx):
        dl = self.get_dl(mode, dl_idx)
        return next(iter(dl))
    


# ## Start experiment 
# Given a maptile, predict its style as one of OSM, CartoVoyager

# In[13]:


from src.models.plmodules.vanilla_vae import VanillaVAE
from src.models.plmodules.iwae import IWAE
from src.models.plmodules.three_fcs import ThreeFCs

from src.data.datamodules.maptiles_datamodule import MaptilesDataModule
from src.data.datamodules.mnist_datamodule import MNISTDataModule


# In[14]:


bs = 128
dm = MNISTDataModule(in_shape=(1,32,32), bs=bs)
dm.setup('fit')


# In[10]:


# dm.train_ds.show_samples(order='chw')


# In[18]:


# Instantiate the pl Module
model_name = 'iwae' # 'vae'
latent_dim = 10
hidden_dims = [32,64,128,256]#,512]
n_samples = 1
act_fn = nn.ReLU()
lr = 1e-3

def models(model_name: str):
    return {'vae': VanillaVAE(in_shape=dm.size(), #dm.in_shape, 
                        latent_dim=latent_dim,
                        hidden_dims=hidden_dims,
                        act_fn=act_fn,
                        learning_rate=lr),
            'iwae': IWAE(
                in_shape=dm.size(), #dm.in_shape, 
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                n_samples=n_samples,
                act_fn=act_fn,
                learning_rate=lr
            )}[model_name]

model = models(model_name)
# print(dm.hparams)
print(model.hparams)


# In[ ]:


# Instantiate a PL `Trainer` object
# -- most basic trainer: uses good defaults, eg: auto-tensorboard logging, checkpoints, logs, etc.
# -- Pass the data module along with a pl module
# ref: https://www.learnopencv.com/tensorboard-with-pytorch-lightning/
# Callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
exp_name = f"{model.name}_{dm.name}"
tb_logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs', name='vae_mnist')
trainer_config = {
#     'fast_dev_run': 7, # for debugging only
    'gpus':1,
    'max_epochs': 100,
    'progress_bar_refresh_rate':20,
#     'auto_lr_find': True,
    'terminate_on_nan':True,
#     'num_sanity_val_steps':0.25,
    'check_val_every_n_epoch':10,
    'logger':tb_logger,
    'profiler': "simple",
#     'callbacks':[EarlyStopping('val_loss')]
}
trainer = pl.Trainer(**trainer_config)
# trainer.tune(model)
# Fit model
trainer.fit(model, dm)


# In[ ]:


# Finally,
# Log this model's hyperparmeters to tensorboard
# hparams = dict(model.hparams)
# metrics = {'hparam/acc': model.hparams["loss"]}
# model.logger.experiment.add_hparams(hparam_dict=hparams,
#                                     metric_dict=metrics) #how to store the 'best' value of the metric?
# Alternatively, use pl.Logger's method "log_hyperparameters"
#         logger.log_hyperparams(hparams, metrics)


# ## Evaluation
# 1. Reconstructions
#     - Given x from train/val/test dataset, show N (eg. 16) number of possible reconstruction
#     - Workflow: 
#         - x --> model.encoder(x) --> theta_z --> sample N latent codes from the Pr(z; theta_z) --> model.decoder(z) for each sampled z's 
# 2. Inspect the topology/landscape of the learned latent space
#     - Latent traversal: Pick a dimension of the latent space. Keep all other dimensions' values constant. Vary the chosen dimenion's values (eg. linearly, spherically) and decode the latent codes. Show the outputs of the decoder.
#     
# 3. Mutual information
#     - Between x and x_sample for N number of x_samples.
#     - Between each dimensions of a latent code
#     

# In[ ]:


from torch.distributions import Normal


# In[ ]:


llhs = {}
dim_x = model.input_dim()
with torch.no_grad():
    for mode in ['train', 'val']:
        prob_sum = torch.tensor(0.)
        n_imgs = 0
        dl = getattr(dm, f"{mode}_dataloader")()
        for (x,y) in dl:           
            bs = x.shape[0]
            
            # sample z -> decoder -> mu_x
            z_sample = torch.randn((bs, model.latent_dim), device=model.device)
            mu_x = model.decode(z_sample)
            
            dist = Normal(mu_x, torch.ones(mu_x.shape))
            breakpoint()
            log_prob = torch.sum(dist.log_prob(x), dim=(1,2,3)) # (bs,1)
            prob = log_prob.exp()
            
            # Accumulate the sum over this batch
            prob_sum += torch.sum(prob)
            prob_per_dim_sum += torch.sum(prob) / dim_x
            n_imgs += bs

        # Log of average likelihood of an image
        llhs[mode] = (prob_sum/n_imgs).item()
        llhs_dim[mode] = (prob_sum/n_imgs).item() 
        
        
        


# In[ ]:


pprint(llhs)


# ### Recons of inputs from training data

# In[ ]:


model.eval();


# In[ ]:


with torch.no_grad():
    for mode in ['train', 'val']:
        dl = getattr(model, f"{mode}_dataloader")()
        x,y = next(iter(dl))
        x_recon = model.generate(x)
        show_timgs(x.detach(), title=f"{mode} dataset")
        show_timgs(x_recon.detach(), title=f"{mode}: recon")


# In[ ]:


with torch.no_grad():
    mu,log_var = model.encode(x)
    f, ax = plt.subplots(1,2)
    ax = ax.flatten()
    ax[0].hist(mu, label='mu')
    ax[0].set_title('mu')
    ax[1].hist(log_var.exp(), label='var')
    ax[1].set_title('var')
    


# In[ ]:


min_var = 1e-4
n_tiny_vars = (log_var.exp() < min_var).sum()
n_tiny_vars, n_tiny_vars/log_var.numel()


# ### Recons of samples from learned latent space

# In[ ]:


n_samples = 36
with torch.no_grad():
    sampled_recons = model.sample(n_samples, model.device)
    show_timgs(sampled_recons.detach())


# ## Inspect latent space
# For each x, first 
# mu, logvar= model.encoder(x)
# z_samples = model.reparametrize(mu, logvar)
# 
# - Compute pairwise distances between each image in  the (mini) batch of input images based on the 
# Show $K$ number of nearest neighbors  analysis
# 

# ## Latent Space Traversal
# 1. Linear traversal in a single dimension

# In[ ]:


chosen_dim = 0 # must be in range(latent_dim)
fixed_vec = torch.randn((1, model.latent_dim-1))
fixed_values = fixed_vec.repeat((n_samples,1))
n_samples = 16
zi_min, zi_max = -2,2
varying = torch.linspace(zi_min, zi_max, n_samples).view((-1,1))

varying.shape,fixed_values.shape




# In[ ]:


def construct_from(a_col:torch.Tensor, other_cols:torch.Tensor, ind):
    """
    Make a tensor from a column vector and a matrx containing all the other columns
    by inserting the `onc_column` at the final matrix's `ind`th column.
    """
    assert a_
    n_cols = 1 + 
    out = a_col.new_zeros((


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





# In[ ]:





# ---
# ## Misc experiments
# 

# ### Q: Does `torch`'s `dtype` conversion (eg. my_tensor.to(torch.float64)) keeps the new tensor attached to the original tensor's computational graph?
# Related:
# - `is_leaf`
# - `requires_grad`
# - `retain_grad`: See [doc](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.is_leaf:~:text=Only%20leaf%20Tensors%20will%20have%20their,non%2Dleaf%20Tensors%2C%20you%20can%20use%20retain_grad().)

# In[ ]:


t = torch.ones(1, dtype=torch.float32, requires_grad=True)
t2 = t.to(torch.float64)
# t2.retain_grad()
print(t.requires_grad, t2.requires_grad)
print(t.is_leaf, t2.is_leaf)


# In[ ]:


t2


# In[ ]:


out = 2*t2**3
out.backward()


# In[ ]:


t2.grad, t.grad


# So, yes, the gradient flows via the tensor generated from the original tensor (`t`) with `.to` operation. Therefore, we conclude the tensor generated from `.to` method remains attached to the orignal tensor's computational graph and acts as a medium (ie. a non-leaf node) through which downstream operation's gradient can flow through to be accumulated at the original tensor `t`'s `.grad` attribute. 
# 
# Unless I want to look at the `.grad` of the derived tensor (`t2`), I don't need to call `.retain_grad()` method on `t2`.

# In[ ]:




