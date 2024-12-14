# python make_sbatch.py --time 48 --gb 128 --bash_files bash_commands/1212_2_train_vda_short_lesslr.sh
# extra file because it has to be 128gb

# action frame short
python src/main_action_frame_short.py experiment_name=main_action_frame_short diffusion_model.training.optimizer.base_lr=3.90625e-7

# not as good as normal lr, killed
# Submitted batch job 54692577