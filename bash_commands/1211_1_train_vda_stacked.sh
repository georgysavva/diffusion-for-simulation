# python make_sbatch.py --time 48 --bash_files bash_commands/1211_1_train_vda_stacked.sh

# # action frame stacked
# python src/main_action_frame_stacked.py experiment_name=main_action_frame_stacked

# action frame stacked xl
python src/main_action_frame_stacked_xl.py experiment_name=main_action_frame_stacked_xl

# action frame stacked xl short
python src/main_action_frame_stacked_xl_short.py experiment_name=main_action_frame_stacked_xl_short

# action frame stacked xl pretrained
python src/main_action_frame_stacked_xl_pretrained.py experiment_name=main_action_frame_stacked_xl_pretrained

# action frame stacked xl pretrained short
python src/main_action_frame_stacked_xl_pretrained_short.py experiment_name=main_action_frame_stacked_xl_pretrained_short

# Submitted batch job 54650582 <- first job, killed
# Submitted batch job 54650583
# Submitted batch job 54650584
# Submitted batch job 54650585
# Submitted batch job 54650586