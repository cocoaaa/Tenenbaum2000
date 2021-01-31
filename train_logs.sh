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
# Whenever enc is a resnet, hidden_dims (of the encoder) is is [32,32,...]
# ie. repeats the first dimension twice

#1. enc-dec: conv-conv
nohup python train.py --model_name="beta_vae" \
--enc_type "conv" --dec_type "conv" \
--latent_dim=10 --hidden_dims 32 64 128 256 \
--kld_weight=1.0  --use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &

# 2. enc-dec: conv-resnet
nohup python train.py --model_name="beta_vae" \
--enc_type "conv" --dec_type "resnet" \
--latent_dim=10 --hidden_dims 32 64 128 256 \
--kld_weight=1.0  --use_beta_scheduler \
--data_name="mnist" --data_root='/data/hayley-old/Tenanbaum2000/data' \
-lr 3e-4 -bs 32 \
--gpu_id=1 --max_epochs=200   --terminate_on_nan=True  \
--log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-30/" &

# 3. enc-dec: resnet-conv

# 4. enc-dec: resnet-resnet