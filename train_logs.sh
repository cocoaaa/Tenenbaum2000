#!/bin/bash

#wm
latent_dims = #todo: as python lang. instead ofsh
nohup python src/train.py --model_name="iwae" --data_name="mnist" --latent_dim=10 --n_sample=200
nohup python src/train.py --model_name="iwae" --data_name="mnist" --latent_dim=10 --n_sample=500

nohup python src/train.py --model_name="iwae" --data_name="maptiles" --latent_dim=10 --n_sample=50
nohup python src/train.py --model_name="iwae" --data_name="maptiles" --latent_dim=10 --n_sample=100
nohup python src/train.py --model_name="iwae" --data_name="maptiles" --latent_dim=10 --n_sample=200
nohup python src/train.py --model_name="iwae" --data_name="maptiles" --latent_dim=10 --n_sample=300

nohup python src/train.py --model_name="vae" --data_name="maptiles" --latent_dim=10
nohup python src/train.py --model_name="vae" --data_name="maptiles" --latent_dim=10
nohup python src/train.py --model_name="vae" --data_name="maptiles" --latent_dim=10

nohup python src/train.py --model_name="iwae" --data_name="maptiles" \
--latent_dim=10 --n_sample=300 --log_root='./temp-logs/'

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-3 --adv_weight 45.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1 --max_epochs=200 --batch_size=128 --terminate_on_nan=True \
    --fast_dev_
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-3 --adv_weight 45.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1 max_epochs=200 --batch_size=128 --terminate_on_nan=True

# Running these
#local(3)
python src/train.py --model_name="iwae" --data_name="mnist" \
--latent_dim=10 --n_sample=10 --max_epochs=30 --gpu_id=1 \
--hist_epoch_interval=1 --hist_epoch_interval=1

#local(4)
python src/train.py --model_name="iwae" --data_name="maptiles" \
--latent_dim=10 --n_sample=10 --max_epochs=30 --gpu_id=1 \
--hist_epoch_interval=1 --hist_epoch_interval=1


python src/train.py --model_name="iwae" --data_name="maptiles" \
 --latent_dim=10 --n_sample=100 --max_epochs=30 --gpu_id=1


python src/train.py --model_name="iwae" --data_name="maptiles" \
 --latent_dim=10 --n_sample=100 --max_epochs=100 --gpu_id=1 \
 --hist_epoch_interval=1 --hist_epoch_interval=1

 python src/train.py --model_name="vae" --data_name="maptiles" \
 --latent_dim=10 --max_epochs=100 --gpu_id=1 \
 --hist_epoch_interval=1 --hist_epoch_interval=1



## Jan 9, 2021
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-3 --adv_weight 45.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1 --max_epochs=200 --batch_size=128 --terminate_on_nan=True


nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-3 --adv_weight 125.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1 --max_epochs=200 --batch_size=128 --terminate_on_nan=True

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-3 --adv_weight 375.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1 --max_epochs=200 --batch_size=128 --terminate_on_nan=True

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-3 --adv_weight 1500.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1 --max_epochs=200 --batch_size=128 --terminate_on_nan=True

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-3 --adv_weight 5.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1 --max_epochs=200 --batch_size=128 --terminate_on_nan=True

#lr for adv weight =5.0 on gpu2
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-3 --adv_weight 5.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 3e-3 --adv_weight 5.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True  &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 9e-3 --adv_weight 5.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 2e-2 --adv_weight 5.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 6e-2 --adv_weight 5.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-1 --adv_weight 5.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True &



#lr for adv weight =15.0 on gpu2
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 3e-4 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-3 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 3e-3 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True  &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 9e-3 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 2e-2 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 6e-2 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True &

#nohup python train_bivae.py --model_name="bivae" \
#--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-1 --adv_weight 15.0 \
#--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
#--gpu_id=2 --max_epochs=200 --batch_size=128 --terminate_on_nan=True &


# Jan 10, 2021
# Run for longer epochs (400)
# First with adv_loss_weight = 5.0
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 3e-4 --adv_weight 5.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=400 --batch_size=128 --terminate_on_nan=True  \
--log_root="./lightning_logs/2021-01-10/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-3 --adv_weight 5.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=400 --batch_size=128 --terminate_on_nan=True  \
--log_root="./lightning_logs/2021-01-10/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 3e-3 --adv_weight 5.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=400 --batch_size=128 --terminate_on_nan=True \
--log_root="./lightning_logs/2021-01-10/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 2e-2 --adv_weight 5.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=400 --batch_size=128 --terminate_on_nan=True \
--log_root="./lightning_logs/2021-01-10/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 6e-2 --adv_weight 5.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=400 --batch_size=128 --terminate_on_nan=True \
--log_root="./lightning_logs/2021-01-10/" &

# W adv_loss_weight = 15.0
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 3e-4 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=400 --batch_size=128 --terminate_on_nan=True  \
--log_root="./lightning_logs/2021-01-10/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 1e-3 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=400 --batch_size=128 --terminate_on_nan=True  \
--log_root="./lightning_logs/2021-01-10/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 3e-3 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=400 --batch_size=128 --terminate_on_nan=True \
--log_root="./lightning_logs/2021-01-10/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 2e-2 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=400 --batch_size=128 --terminate_on_nan=True \
--log_root="./lightning_logs/2021-01-10/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 -lr 6e-2 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=400 --batch_size=128 --terminate_on_nan=True \
--log_root="./lightning_logs/2021-01-10/" &

# ------------------------------------------------------------------------
# Jan 11, 2021
# Run traning with lr monitoring callback and an lr-scheduler (plateau)
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 -lr 1e-3  --terminate_on_nan=True  \
--log_root="./lightning_logs/2021-01-11/" &

# Run w/o adv_loss_c
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--adv_weight 15.0  --not_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --terminate_on_nan=True  \
--max_epochs=200 --batch_size=128 -lr 1e-3 \
--log_root="./lightning_logs/2021-01-11/" &

# ------------------------------------------------------------------------
# Jan 12, 2021
# Hyperparameter search for BiVAE (w/o contrasive loss)
# - learning_rate: [3e-4, 1e-3, 3e-3, 2e-2, 6e-2]
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--adv_weight 15.0  --not_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --terminate_on_nan=True  \
--max_epochs=200 --batch_size=128 -lr 3e-4 \
--log_root="./lightning_logs/2021-01-12/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--adv_weight 15.0  --not_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --terminate_on_nan=True  \
--max_epochs=200 --batch_size=128 -lr 1e-3 \
--log_root="./lightning_logs/2021-01-12/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--adv_weight 15.0  --not_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --terminate_on_nan=True  \
--max_epochs=200 --batch_size=128 -lr 3e-3 \
--log_root="./lightning_logs/2021-01-12/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--adv_weight 15.0  --not_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --terminate_on_nan=True  \
--max_epochs=200 --batch_size=128 -lr 2e-2 \
--log_root="./lightning_logs/2021-01-12/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--adv_weight 15.0  --not_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --terminate_on_nan=True  \
--max_epochs=200 --batch_size=128 -lr 6e-2 \
--log_root="./lightning_logs/2021-01-12/" &


#adv_weight w/ fixed lr=1e-3, bs=128
# adv_weight: [5., 15., 45., 135, 405, 1215.]
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--adv_weight 5.0  --not_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --terminate_on_nan=True  \
--max_epochs=200 --batch_size=128 -lr 1e-3 \
--log_root="./lightning_logs/2021-01-12/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--adv_weight 15.0  --not_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --terminate_on_nan=True  \
--max_epochs=200 --batch_size=128 -lr 1e-3 \
--log_root="./lightning_logs/2021-01-12/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--adv_weight 45.0  --not_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --terminate_on_nan=True  \
--max_epochs=200 --batch_size=128 -lr 1e-3 \
--log_root="./lightning_logs/2021-01-12/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--adv_weight 135.0  --not_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --terminate_on_nan=True  \
--max_epochs=200 --batch_size=128 -lr 1e-3 \
--log_root="./lightning_logs/2021-01-12/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--adv_weight 405.0  --not_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --terminate_on_nan=True  \
--max_epochs=200 --batch_size=128 -lr 1e-3 \
--log_root="./lightning_logs/2021-01-12/" &

nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--adv_weight 1215.0  --not_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --terminate_on_nan=True  \
--max_epochs=200 --batch_size=128 -lr 1e-3 \
--log_root="./lightning_logs/2021-01-12/" &

# Hyperparmeter tuning with Ray
nohup python tune_hparams_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 15.0 \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=200 --batch_size=128 -lr 1e-3  --terminate_on_nan=True  \
--log_root="./lightning_logs/2021-01-12/" &

# 2021-1-13
# Hyperparmeter tuning with latent_dim = [16, 32, 64]

# 2121-1-14
#Use beta annealing scheduler
 nohup python tune_hparams_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 15.0 \
--use_beta_scheduler \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=2 --max_epochs=300 --batch_size=128 -lr 1e-3  --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-14-ray/" &

# 2021-1-16
# Train BiVAE-C
# with Beta annealing scheduler
# and KLD of the content/style latent subspace tracking
# log_dir = '/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-16/BiVAE-C_MNIST-red-green-blue_seed-123/version_1'
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 --adv_weight 15.0  --is_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1 --max_epochs=300 --batch_size=32 -lr 1e-3 --terminate_on_nan=True  \
--use_beta_scheduler \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-16/" &

# To visualize the clusters in content space vs. color-coding by style labels or by content labels
# (and repeat the same evaluation on clutering based on style space and on whole z code
# we set latent dim to be 4, so that each partition space can be visualized without any dimension-reduction
# algorithm (eg. tSNE)
# log_dir = '/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-16/BiVAE-C_MNIST-red-green-blue_seed-123/version_0'
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=4 --hidden_dims 32 64 128 256 --adv_dim 32 32 --adv_weight 15.0  --is_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1 --max_epochs=300 --batch_size=32 -lr 1e-3 --terminate_on_nan=True  \
--use_beta_scheduler \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-16/" &

# Q: what would be a good balance btw kld_weight vs. adv_weight (w/o beta annealing)
# Run Hyperparameter-search
# - with kld_weight added to the search space
# - w/ or w/o kld_weight annealing (ie. BetaScheduler)
# Fianl Search space:
# is_contrasive, kld_weight, use_beta_scheduler, adv_loss_weight, learning_rate, batch_size
 nohup python tune_loss_weights_bivae.py --model_name="bivae" \
 --latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
 --data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
 --gpu_id=2 --max_epochs=300   --terminate_on_nan=True  \
 --log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-16-ray/" &


# 2021-1-18
# Followup of 2021-1-16 Runs are using 32 32  adv_dim. Here we rerun the code with 32 32 32 for consistency with other exps
# Train BiVAE-C
# --with Beta annealing scheduler
# --and KLD of the content/style latent subspace tracking
# log_dir = '/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-18/BiVAE-C_MNIST-red-green-blue_seed-123/version_0'
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 15.0  --is_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1 --max_epochs=300 --batch_size=32 -lr 1e-3 --terminate_on_nan=True  \
--use_beta_scheduler \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-18/" &

# To visualize the clusters in content space vs. color-coding by style labels or by content labels
# (and repeat the same evaluation on clutering based on style space and on whole z code
# we set latent dim to be 4, so that each partition space can be visualized without any dimension-reduction
# algorithm (eg. tSNE)
# log_dir = '/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-18/BiVAE-C_MNIST-red-green-blue_seed-123/version_1'
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=4 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 15.0  --is_contrasive \
--data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
--gpu_id=1 --max_epochs=300 --batch_size=32 -lr 1e-3 --terminate_on_nan=True  \
--use_beta_scheduler \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-18/" &

# 2021-1-20
# tune_loss_weights_bivae.py
# Given a specific model architecture and a fixed-learning rate(?), search for the best balance of the loss
# components.
# Fixed latent_dim =10, and model acchitecture (hidden_dims, adv_dim)
# And, the hparam search space is:
# --is_contrasive
# --kld_weight
# --adv_weight
# --use_beta_scheduler: if set to True, use the cyclic linear scheduler with the max_kld set to the `kld_weight`
# --learning_rate
# --batch_size
 nohup python tune_loss_weights_bivae.py --model_name="bivae" \
 --latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
 --data_name="multi_mono_mnist" --colors red green blue --n_styles=3 \
 --gpu_id=1 --max_epochs=300   --terminate_on_nan=True  \
 --log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-20-ray/" &

# Jan 23, 2021
# Train BiVAE on Multi Rotated MNIST with angles = [-45, 0, 45]
 nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--data_name="multi_rotated_mnist" --angles -45 0 45 --n_styles=3 \
--gpu_id=2 --max_epochs=400   --terminate_on_nan=True  \
-lr 3e-4 --adv_weight 15.0 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23/" &

nohup python tune_loss_weights_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 \
--data_name="multi_rotated_mnist" --angles -45 0 45 --n_styles=3 \
--gpu_id=1 --max_epochs=400   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23-ray/" &

# Train BiVAE on Multi Maptiles with
#data_root = Path("/data/hayley-old/maptiles_v2/")
#cities = ['berlin', 'rome'] #['la', 'paris']
#CartoVoyagerNoLabels", "StamenTonerBackground"
#styles = ['OSMDefault', 'CartoVoyagerNoLabels']
#styles = ["CartoVoyagerNoLabels", "StamenTonerBackground"]
#zooms = ['14']
#in_shape = (3,32,32)
#batch_size = 32
nohup python train_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 15.0 \
--data_name="multi_maptiles" \
--cities la paris \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=3 \
--zooms 14 \
--gpu_id=2 --max_epochs=400   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23/" &

nohup python tune_loss_weights_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities la paris \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=3 \
--zooms 14 \
--gpu_ids=2 --max_epochs=400   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23-ray/" &

#cities = ['berlin', 'rome']
nohup python tune_loss_weights_bivae.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities berlin rome \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=3 \
--zooms 14 \
--gpu_ids 1 2 --max_epochs=400   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23-ray/" &

# ------------------------------------------------------------------------
# 2021-1-30
# Test BetaVAE with conv/resnet encoder -- conv/resnet decoder pair on MNIST
# with beta=1.0 + Beta cyclic scheduler
# Whenever enc is a resnet, hidden_dims (of the encoder) is [32,32,...]
# ie. repeats the first dimension twice


#1. enc-dec: conv-conv
# BetaVAE-conv-conv-1.000_MNIST/version0
nohup python train.py --model_name="beta_vae" \
--enc_type "conv" --dec_type "conv" \
--latent_dim=10 --hidden_dims 32 64 128 256 \
--kld_weight=1.0  --use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &

# 2.1 enc-dec: conv-resnet
# BetaVAE-conv-resnet-1.000_MNIST/version0
nohup python train.py --model_name="beta_vae" \
--enc_type "conv" --dec_type "resnet" \
--latent_dim=10 --hidden_dims 32 32 64 128 256 \
--kld_weight=1.0  --use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &

# 2.2 enc-dec: conv-resnet
# BetaVAE-conv-resnet-1.000_MNIST/version1
nohup python train.py --model_name="beta_vae" \
--enc_type "conv" --dec_type "resnet" \
--latent_dim=10 --hidden_dims 32  64 128 256 \
--kld_weight=1.0  --use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &

# 3. enc-dec: resnet-conv
# BetaVAE-resnet-conv-1.000_MNIST/version0
nohup python train.py --model_name="beta_vae" \
--enc_type "resnet" --dec_type "conv" \
--latent_dim=10 --hidden_dims 32 32 64 128 256 \
--kld_weight=1.0  --use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &

# 4. enc-dec: resnet-resnet
# BetaVAE-resnet-resnet-1.000_MNIST/version0
nohup python train.py --model_name="beta_vae" \
--enc_type "resnet" --dec_type "resnet" \
--latent_dim=10 --hidden_dims 32 32 64 128 256 \
--kld_weight=1.0  --use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &

# Run the above 4 kinds of beta-VAE (with each enc-dec type pair) without beta annealing scheduler
# --not_use_beta_scheduler
# 1. enc-dec: conv-conv
# BetaVAE-conv-conv-1.000_MNIST/version1
nohup python train.py --model_name="beta_vae" \
--enc_type "conv" --dec_type "conv" \
--latent_dim=10 --hidden_dims 32 64 128 256 \
--kld_weight=1.0  --not_use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &

# 2.1 enc-dec: conv-resnet
# BetaVAE-conv-resnet-1.000_MNIST/version2
nohup python train.py --model_name="beta_vae" \
--enc_type "conv" --dec_type "resnet" \
--latent_dim=10 --hidden_dims 32 32 64 128 256 \
--kld_weight=1.0  --not_use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &

# 2.2 enc-dec: conv-resnet
# BetaVAE-conv-resnet-1.000_MNIST/version3
nohup python train.py --model_name="beta_vae" \
--enc_type "conv" --dec_type "resnet" \
--latent_dim=10 --hidden_dims 32  64 128 256 \
--kld_weight=1.0  --not_use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &

# 3. enc-dec: resnet-conv
# BetaVAE-resnet-conv-1.000_MNIST/version1
nohup python train.py --model_name="beta_vae" \
--enc_type "resnet" --dec_type "conv" \
--latent_dim=10 --hidden_dims 32 32 64 128 256 \
--kld_weight=1.0  --not_use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &

#(oops, mistake - ran without changing the beta scheduler flag)
# 4.1 enc-dec: resnet-resnet
# BetaVAE-resnet-resnet-1.000_MNIST/version1
# Oops, I ran with the wrong flay for the beta scheduler on this one.
nohup python train.py --model_name="beta_vae" \
--enc_type "resnet" --dec_type "resnet" \
--latent_dim=10 --hidden_dims 32 32 64 128 256 \
--kld_weight=1.0  --use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &

# With the correct flag:
# 4.2 enc-dec: resnet-resnet
# BetaVAE-resnet-resnet-1.000_MNIST/version2
nohup python train.py --model_name="beta_vae" \
--enc_type "resnet" --dec_type "resnet" \
--latent_dim=10 --hidden_dims 32 32 64 128 256 \
--kld_weight=1.0  --not_use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &


# For each enc-dec pair, search the beta (kld_weight) space (w/ or w/o beta_scheduler)
# search_space = {
#    "latent_dim": 10, #tune.grid_search([16, 32, 64,128]),
#    'kld_weight': tune.grid_search([0., 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32., 64, 128., 256, 512, 1024]),
#    'use_beta_scheduler': tune.grid_search([False,True]),
#    'learning_rate': tune.grid_search(list(np.logspace(-4., -1, num=10))),
#    'batch_size': tune.grid_search([32, 64, 128]),
#}
# 1. enc-dec: conv-conv
nohup python tune_hparams_vae.py --model_name="beta_vae" \
--enc_type "conv" --dec_type "conv" \
--latent_dim=10 --hidden_dims 32 64 128 256 \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
--gpu_id=2 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30-ray/" &

# 2. enc-dec: conv-resnet
nohup python tune_hparams_vae.py --model_name="beta_vae" \
--enc_type "conv" --dec_type "resnet" \
--latent_dim=10 --hidden_dims 32 64 128 256 \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
--gpu_id=2 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30-ray/" &

# 3. enc-dec: resnet-conv
nohup python tune_hparams_vae.py --model_name="beta_vae" \
--enc_type "resnet" --dec_type "conv" \
--latent_dim=10 --hidden_dims 32 32 64 128 256 \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
--gpu_id=2 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30-ray/" &

# 4. enc-dec: resnet-resnet
nohup python tune_hparams_vae.py --model_name="beta_vae" \
--enc_type "resnet" --dec_type "resnet" \
--latent_dim=10 --hidden_dims 32 32 64 128 256 \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
--gpu_id=2 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30-ray/" &

# Train beta vae with maptiles with varying hyperparams
# kld_weight
# Eventaully stop after today and look at embedding etc
# Not interested in the recon quality
# - that's why we don't use GAN

# I ran a bunch of training of Beta-VAE with different (enc-dec) pairs on Maptiles in Jupyter Notebook
# See the notebooks, naned "14-..."
# Log Dir Example:
# /data/hayley-old/Tenanbaum2000/temp-logs/
# BetaVAE-resnet-resnet-1.000_Maptiles_la-charlotte-vegas-boston-paris-amsterdam-shanghai-seoul-chicago-manhattan-berlin-montreal-rome_StamenTonerBackground_14/version_1

# cities = all cities (la-charlotte-vegas-boston-paris-amsterdam-shanghai-seoul-chicago-manhattan-berlin-montreal-rome)
# One style = StamenTonerBackground
# Zoom = 14

# As above, beta is fixed to 1.0, And, used Cyclic Beta annealing


#------------------------------------------------------------------------
# Mar 1, 2021
# Hparam search: BiVAE with different settings of (encType, decType) on a single style Maptiles
# -- One search per (encType, decType)
# -- Main goal is to search over the weights of loss components (ie. kld_weight, adv_loss_weight)
# -- We also search over both with or w/o beta(kld_weight) cyclic scheduler
# Dataset params
# cities =
# styles =
# zooms =
# in_shape =
# batch_size = 32

# First: with two styles
nohup python train.py --model_name="bivae" \
--enc_type "conv" --dec_type "resnet" \
--latent_dim=10 --hidden_dims 32 64 128 256 512 --adv_dim 32 32 32 --adv_weight 100.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--gpu_id=2 --max_epochs=200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-01/" &

# Hyperparmeter tuning with Ray
# -- Search over: latent_dim, enc_type, dec_type, is_contrasive_kld_weight, adv_loss_weight, lr, bs
# -- Fixed: no beta scheduler
# -- Initial run failed to start (due to memory lack ? ) on march 01
# -- Rerunning on March 4, 2021
nohup python tune_hparams_bivae.py --model_name="bivae" \
--enc_type 'conv' --dec_type 'resnet' \
--latent_dim 20 \
--hidden_dims 32 64 128 256 512 --adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--gpu_id=0 --max_epochs=200  --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-04-ray/" \ &

#----
# Second: with 3 styles
nohup python train.py --model_name="bivae" \
--latent_dim=10 --hidden_dims 32 64 128 256 512 --adv_dim 32 32 32 --adv_weight 100.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--gpu_id=2  --max_epochs=200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-01/" &

# Hparam search: BiVAE with different settings of (encType, decType) on a Multi-style Maptiles
# -- One search per (encType, decType)
# -- Main goal is to search over the weights of loss components (ie. kld_weight, adv_loss_weight)
# -- We also search over both with or w/o beta(kld_weight) cyclic scheduler
# -- Initial run failed to start (due to memory lack ? ) on march 01
# -- Rerunning on March 4, 2021
nohup python tune_hparams_bivae.py --model_name="bivae" \
--enc_type 'conv' --dec_type 'resnet' \
--latent_dim 20 \
--hidden_dims 32 64 128 256 512 --adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--gpu_id=1 --max_epochs=200  --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-04-ray/" \
 | tee -a log.out &




#------------------------------------------------------------------------
# Mar 4, 2021
# in shape = (3,64,64)
# Started to run it 6:20pm Mar 4, 2021
nohup python train.py --model_name="bivae" \
--enc_type "conv" --dec_type "resnet" \
--latent_dim=10 --hidden_dims 32 64 128 256 512 --adv_dim 32 32 32 --adv_weight 100.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id=2 --max_epochs=200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-04/" &

# 2 tiles; size (3,64,64)
# Started to run it 6:34pm Mar 4, 2021
nohup python tune_hparams_bivae.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim 20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground \
--n_styles=2 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_ids 0 --max_epochs 100  --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-04-ray/" \ &




# 3styles
nohup python train.py --model_name="bivae" \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--adv_weight 100.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-04/" &



nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--adv_weight 100.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-04/" &






nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--adv_weight 50.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-04/" &


nohup python train.py \
--model_name="bivae" \
--enc_type 'resnet' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--adv_weight 50.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-04/" &


nohup python train.py \
--model_name="bivae" \
--enc_type 'resnet' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--adv_weight 100.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-04/" &

nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--adv_weight 200.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-04/" &

#------------------------------------------------------------------------
#Mar 5, 2021
## 3 styles
## KLD_WEIGHT = 10; varying adv_weight
#  PID:  10772
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--adv_weight 50.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &


## 16828
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--adv_weight 100.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## 17210
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--adv_weight 150.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

##17653
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--adv_weight 300.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## 18055
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--adv_weight 500.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &


## Same suits as above, but without beta scheduler
##  --not_use_beta_scheduler \
#  PID: 22906
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--not_use_beta_scheduler \
--adv_weight 50.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &


## 23271
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--not_use_beta_scheduler \
--adv_weight 100.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## 23633
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim 20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--not_use_beta_scheduler \
--adv_weight 150.0 \
--data_name "multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles 3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

##  23797
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--not_use_beta_scheduler \
--adv_weight 300.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## 23950
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--not_use_beta_scheduler \
--adv_weight 500.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &



## GPU 1
## kld_weight = 1.0; varying adv_weight
#  PID:24600
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 1.0 \
--adv_weight 50.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 1  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &


##  25362
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 1.0 \
--adv_weight 100.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 1  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

##  25562
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 1.0 \
--adv_weight 150.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 1  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## 25680
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 1.0 \
--adv_weight 300.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 1  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

##  25932
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 1.0 \
--adv_weight 500.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 1  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &



## GPU 1
## kld_weight = 30.0; varying adv_weight
#  PID: 27694
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 30.0 \
--adv_weight 50.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 1  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &


## 28129
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 30.0 \
--adv_weight 100.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 1  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## 28253
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 30.0 \
--adv_weight 150.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 1  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## 28348
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 30.0 \
--adv_weight 300.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 1  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## 28450
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 30.0 \
--adv_weight 500.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 1  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

### 2 STYLES
## GPU 0
## kld_weight = 1.0; varying adv_weight
#  PID:  1297
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 1.0 \
--adv_weight 50.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 0  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &


##  1420
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 1.0 \
--adv_weight 100.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 0  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## 1625
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 1.0 \
--adv_weight 150.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 0  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## 1723
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 1.0 \
--adv_weight 300.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 0  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## 1955
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 1.0 \
--adv_weight 500.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 0  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &


## kld_weight = 10.0; varying adv_weight
#  PID: 2451
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--adv_weight 50.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 0  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## gpu1
## 2692
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--adv_weight 100.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 1  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## gpu2
##  2904
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--adv_weight 150.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 2  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

##  4117
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--adv_weight 300.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 0  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## 4202
nohup python train.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--kld_weight 10.0 \
--adv_weight 500.0 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 0  --max_epochs 200   --terminate_on_nan=True  \
-lr 3e-4 -bs 32 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-05/" &

## Mar 6, 2021
# Hyperparmeter tuning with Ray
# -- Search over: latent_dim, enc_type, dec_type, is_contrasive_kld_weight, adv_loss_weight, lr, bs
# -- Fixed: no beta scheduler
# -- Initial run failed to start (due to memory lack ? ) on march 01
# -- Rerunning on March 4, 2021
#--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \

# PID: 12963
nohup python tune_hparams_bivae.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim 20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--gpu_ids 1 --max_epochs 200  --terminate_on_nan=True  \
--n_ray_samples 20 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-06-ray/" \ &

# 14535
nohup python tune_hparams_bivae.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim 20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--gpu_ids 1 --max_epochs 200  --terminate_on_nan=True  \
--n_ray_samples 20 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-06-ray/" \ &


# 4398
nohup python tune_hparams_bivae.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim 20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault StamenWatercolor --n_styles=4 \
--zooms 14 \
--n_ray_samples 1 \
--gpu_ids 0 --max_epochs 200  --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-06-ray/" \ &


# Ray PBT
nohup python tune_bivae_pbt.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'resnet' \
--latent_dim 20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--gpu_ids 1 --max_epochs 200  --terminate_on_nan=True  \
--n_ray_samples=20 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-06-ray/" \ &




nohup python tune_bivae_pbt.py \
--model_name="bivae" \
--enc_type 'resnet' \
--dec_type 'resnet' \
--latent_dim 20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=2 \
--zooms 14 \
--gpu_ids 2 --max_epochs 200  --terminate_on_nan=True  \
--n_ray_samples 20 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-06-ray/" \ &

# 4 styles!
nohup python tune_bivae_pbt.py \
--model_name="bivae" \
--enc_type 'conv' \
--dec_type 'conv' \
--latent_dim 20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault StamenWatercolor --n_styles=4 \
--zooms 14 \
--gpu_ids 0 --max_epochs 200  --terminate_on_nan=True  \
--n_ray_samples 20 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-06-ray/" \ &




# Mar 8, 2021
# Fixed tune's search space from `grid_search` to `sampling types in order to run
# multiple workers by setting `n_ray_samples` to > 1
nohup python tune_bivae.py \
--model_name="bivae" \
--latent_dim 20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32  \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles StamenTonerBackground --n_styles=1 \
--zooms 14 \
--gpu_ids 0 --max_epochs 150  --terminate_on_nan=True  \
--n_ray_samples 1 \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-08-ray/" \ &


nohup python train_grid_search.py \
--model_name="bivae" \
--latent_dim=20 \
--hidden_dims 32 64 128 256 512 \
--adv_dim 32 32 32 \
--data_name="multi_maptiles" \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
          'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' 'berlin' 'montreal' 'rome' \
--styles CartoVoyagerNoLabels StamenTonerBackground OSMDefault --n_styles=3 \
--zooms 14 \
--in_shape 3 64 64 \
--gpu_id 0  --max_epochs 150   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-03-08/" \ &