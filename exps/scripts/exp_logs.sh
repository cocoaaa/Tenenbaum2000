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


