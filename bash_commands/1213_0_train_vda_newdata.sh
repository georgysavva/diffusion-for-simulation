# python make_sbatch.py --time 48 --bash_files bash_commands/1213_0_train_vda_newdata.sh

# action frame biglr
python src/main_action_frame_newdata.py experiment_name=main_action_frame_newdata_biglr diffusion_model.training.optimizer.base_lr=1.5625e-6

# action frame hugelr
python src/main_action_frame_newdata.py experiment_name=main_action_frame_newdata_hugelr diffusion_model.training.optimizer.base_lr=3.125e-6

# Submitted batch job 54763363
# Submitted batch job 54763364