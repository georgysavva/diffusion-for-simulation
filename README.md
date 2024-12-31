# Transformer-Based Diffusion for Game Generation

This project applies transformer-based Diffusion models to simulate Doom (the find my way home scenario). 

See the final report [here](https://georgysavva.github.io/assets/pdf/Diffusion_for_Simulation.pdf).

## Local Setup

1. Create the conda environment

```
conda env create -f env.yml --prefix ./penv
```

2. Activate the conda environment:

 ```
 conda activate ./penv
```

3. Install python dependencies

```
pip install -r requirements.txt
```

4. Install the root package

```
pip install -e .
```

5. Login to wandb

```
wandb login
```

## To Train locally

1. Train the VAE
    1. Collect frames data for VAE training

    ```
    python scripts/collect_doom_data.py --observations_only --save_path {vae_train_data}
   ````

    2. Fine tune the VAE decoder

    ```
    python src/vae/train_vae.py --data_dir {vae_train_data}
    ```

2. Collect game play data using a random policy

    1. Train data

    ```
    python scripts/collect_doom_data.py --save_path {data_dir}/original/train
    ```

    2. Test data

    ```
    python scripts/collect_doom_data.py --save_path {data_dir}/original/test --num_episodes 200
    ```

3. Preprocess gameplay data with the VAE

```
python scripts/preprocess_data_with_vae.py --data_path {data_dir}/original --save_path {data_dir}/latent
```

4. Train the diffusion model. You will need to update  `project_storage_base_path` for the output directory, the vae decoder path, and `data_path` to `{data_dir}` in hydra configs

 - To train the concatenation conditioning single-frame prediction model: 
   ```
   python src/main.py diffusion_model=DiT_B_4_concat experiment_name=DiT_B_4_concat
   ```

 - To train the cross-attention conditioning single-frame prediction model: 
   ```
   python src/main.py diffusion_model=DiT_B_4_cross_attn experiment_name=DiT_B_4_cross_attn
   ```
 
 - To train the video generation multi-frame prediction model: 
   ```
   git checkout vg-model && python src/main.py experiment_name=DiT_VG
   ```

## Inference

To evaluate your single-frame prediction model on a trajectory:
```
python scripts/evaluate_trajectory --run_dir {training_run_dir} --vae_decoder_path {finetuned_decoder} --episode_path {test_episode}
```

To evaluate your video-generation model on a trajectory:
```
git checkout vg-model && python scripts/evaluate_trajectory_vg.py --run_dir {training_run_dir} --model_version {model_file_name} --episodes_path {file_with_episode_paths}
```
