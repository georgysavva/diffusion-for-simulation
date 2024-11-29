# python make_sbatch.py --time 48 --bash_files bash_commands/1128_0_train_vae.sh

# test
# python src/vae/train_vae.py --tag test --learning_rate 1e-5 --max_train_steps 500 --checkpointing_steps 50 --n_eval_times 5 --wandb

# train_vae_3e-6
python src/vae/train_vae.py --tag 1128_0_train_vae_lr3e-6 --learning_rate 3e-6 --wandb

# train_vae_1e-6
python src/vae/train_vae.py --tag 1128_0_train_vae_lr1e-6 --learning_rate 1e-6 --wandb

# train_vae_1e-7
python src/vae/train_vae.py --tag 1128_0_train_vae_lr1e-7 --learning_rate 1e-7 --wandb

# Submitted batch job 54097636
# Submitted batch job 54097637
# Submitted batch job 54097638