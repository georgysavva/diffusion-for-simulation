# python make_sbatch.py --time 48 --bash_files bash_commands/1212_3_train_vda_newdata_lesslr.sh

# action frame
python src/main_action_frame_newdata.py experiment_name=main_action_frame_newdata diffusion_model.training.optimizer.base_lr=3.90625e-7

# action frame xl pretrained
python src/main_action_frame_xl_pretrained_newdata.py experiment_name=main_action_frame_xl_pretrained_newdata diffusion_model.training.optimizer.base_lr=3.90625e-7

# action frame xl pretrained short
python src/main_action_frame_xl_pretrained_short_newdata.py experiment_name=main_action_frame_xl_pretrained_short_newdata diffusion_model.training.optimizer.base_lr=3.90625e-7

# Submitted batch job 54692578
# Submitted batch job 54692579
# Submitted batch job 54692580