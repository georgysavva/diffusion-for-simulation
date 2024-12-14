# python make_sbatch.py --time 48 --gb 128 --bash_files bash_commands/1212_3_train_vda_short_newdata_lesslr.sh
# extra file because it has to be 128gb

# action frame short
python src/main_action_frame_short_newdata.py experiment_name=main_action_frame_short_newdata diffusion_model.training.optimizer.base_lr=3.90625e-7

# Submitted batch job 54692581