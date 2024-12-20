# Diffusion for Game Simulation with Transformers

This project applies Transformer based Diffusion models to simulate Doom (the find my way home scenario).

This repository is based on the [diamond](https://github.com/eloialonso/diamond/tree/csgo) repo. The DiT model code is based on the official DiT [repo](https://github.com/facebookresearch/DiT/blob/main/models.py). We use [hydra](https://hydra.cc/docs/intro/) for configuration.

## Local Setup

1. Create the conda environment

`conda env create -f env.yml --prefix ./penv`

2. Activate the conda environment:

 `conda activate ./penv`

3. Install python dependencies

`pip install -r requirements.txt`

4. Install the root package

`pip install -e .`

5. Login to wandb

`wandb login`

## To Train locally

1. Train the VAE
    1. Collect frames data for VAE training

    `python scripts/collect_doom_data.py --observations_only --save_path {vae_train_data}`

    2. Fine tune the VAE decoder

    `python src/vae/train_vae.py --data_dir {vae_train_data}`

2. Collect game play data using a random policy

    1. Train data

    `python scripts/collect_doom_data.py --save_path {data_dir}/train`

    2. Test data

    `python scripts/collect_doom_data.py --save_path {data_dir}/test --num_episodes 200`

3. Preprocess gameplay data with the VAE

`python scripts/preprocess_data_with_vae.py --data_path {data_dir} --save_path {save_dir}`

4. Train the diffusion model. You will need to update the vae and data paths in the hydra configs

To train the concatenation conditioning single-frame prediction model:
`git checkout main && python src/main.py experiment_name=baseline`

To train the cross-attention conditioning single-frame prediction model:
`git checkout main && python src/main.py experiment_name=baseline`

To train the video generation multi-frame prediction model:
`git checkout jack/vda_model && python src/main_action_frame_newdata.py experiment_name=main_action_frame_newdata`

For each command, make sure to modify the hydra configs parameters `project_storage_base_path` for the output directory and `data_path` for VAE preprocessed data.

## Inference

1. To evaluate your model on a trajectory, for models that predict one frame at a time, run

`python scripts/evaluate_trajectory --run_dir {training_run_dir} --vae_decoder_path {finetuned_decoder} --episode_path {test_episode}`
