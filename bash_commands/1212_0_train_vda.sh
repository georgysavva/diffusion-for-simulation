# python make_sbatch.py --time 48 --bash_files bash_commands/1212_0_train_vda.sh

# action frame
python src/main_action_frame.py experiment_name=main_action_frame

# action frame xl pretrained
python src/main_action_frame_xl_pretrained.py experiment_name=main_action_frame_xl_pretrained

# action frame xl pretrained short
python src/main_action_frame_xl_pretrained_short.py experiment_name=main_action_frame_xl_pretrained_short

# Submitted batch job 54685516
# Submitted batch job 54685517
# Submitted batch job 54685518