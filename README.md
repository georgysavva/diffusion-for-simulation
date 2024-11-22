# Diffusion for Game Simulation with Transformers

This project tries to apply Transformer based Diffusion models to simulate Doom (the find my way home scenario). The trajectories data were generated using a random policy as saved to `/scratch/gs4288/shared/diffusion_for_simulation/data/doom/` on NYU Greene. The code assumes access to that folder.

This repository is based on the [diamond](https://github.com/eloialonso/diamond/tree/csgo) repo. So far it only has the vanilla DiT diffusion model. The model code was taken as is from the official DiT [repo](https://github.com/facebookresearch/DiT/blob/main/models.py). We use [hydra](https://hydra.cc/docs/intro/) for configuration.

## Local Setup

1. Create the conda environment: `conda env create -f env.yml --prefix ./penv`. Whenever you install a system library install it via conda and add the new system dependency to the env config file.
2. Activate the conda environment: `conda activate ./penv`
3. Install python dependencies: `pip install -r requirements.txt`. Whenever you add new python dependencies don't forget to add them to the `requirements.txt` file.
4. Everything is ready!
