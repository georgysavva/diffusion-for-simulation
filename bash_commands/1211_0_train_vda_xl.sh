# python make_sbatch.py --time 48 --bash_files bash_commands/1211_0_train_vda_xl.sh

# action frame xl short
python src/main_action_frame_xl_short.py experiment_name=main_action_frame_xl_short

# action frame xl pretrained short
python src/main_action_frame_xl_pretrained_short.py experiment_name=main_action_frame_xl_pretrained_short

# action frame xl
python src/main_action_frame_xl.py experiment_name=main_action_frame_xl

# action frame xl pretrained
python src/main_action_frame_xl_pretrained.py experiment_name=main_action_frame_xl_pretrained

# Submitted batch job 54650551 xl <- kill
# Submitted batch job 54650552 xl_pretrained <- good but killed because of lengthy eval
# Submitted batch job 54650553 xl_pretrained_short <- good but killed because of lengthy eval
# Submitted batch job 54650554 xl_short <- kill
