# Diffusion for Game Simulation with Transformers

This project applies Transformer based Diffusion models to simulate Doom (the find my way home scenario).

This repository is based on the [diamond](https://github.com/eloialonso/diamond/tree/csgo) repo. The DiT model code is based on the official DiT [repo](https://github.com/facebookresearch/DiT/blob/main/models.py). We use [hydra](https://hydra.cc/docs/intro/) for configuration.

## Local Setup

1. Create the conda environment: `conda env create -f env.yml --prefix ./penv`. Whenever you install a system library install it via conda and add the new system dependency to the env config file.
2. Activate the conda environment: `conda activate ./penv`
3. Install python dependencies: `pip install -r requirements.txt`. Whenever you add new python dependencies don't forget to add them to the `requirements.txt` file.
4. Install the root package `pip install -e .`
5. Login to wandb `wandb login`
6. Everything is ready!

## To Train locally

1. Train the VAE
    1. Collect frames data for VAE training `python scripts/collect_doom_data.py --observations_only --save_path {vae_train_data}`
    2. Fine tune the VAE decoder `python src/vae/train_vae.py --data_dir {vae_train_data}`
2. Collect game play data using a random policy
    1. Train data `python scripts/collect_doom_data.py --save_path {train_data_dir}`
    2. Test data `python scripts/collect_doom_data.py --save_path {test_data_dir} --num_episodes 200`
3. Preprocess game play data with the VAE `python scripts/preprocess_data_with_vae.py --data_path {path_to_folder_with_train_and_test} --save_path {save_dir}`
4. Train the diffusion model `python src/main.py experiment_name=baseline`. You will need to update the vae and data paths in the hydra configs

## Inference

- To evaluate your model on a trajectory, for models that predict one frame at a time, run `python scripts/evaluate_trajectory --run_dir {training_run_dir} --vae_decoder_path {finetuned_decoder} --episode_path {test_episode}`
