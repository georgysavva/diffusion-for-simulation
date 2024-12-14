# python make_sbatch.py --time 48 --bash_files bash_commands/1212_2_train_vda_lesslr.sh

# action frame
python src/main_action_frame.py experiment_name=main_action_frame diffusion_model.training.optimizer.base_lr=3.90625e-7

# action frame xl pretrained
python src/main_action_frame_xl_pretrained.py experiment_name=main_action_frame_xl_pretrained diffusion_model.training.optimizer.base_lr=3.90625e-7

# action frame xl pretrained short
python src/main_action_frame_xl_pretrained_short.py experiment_name=main_action_frame_xl_pretrained_short diffusion_model.training.optimizer.base_lr=3.90625e-7

# not as good as normal lr, killed
# Submitted batch job 54692557
# Submitted batch job 54692558
# Submitted batch job 54692559