# python make_sbatch.py --time 48 --bash_files bash_commands/1212_1_train_vda_newdata.sh

# action frame
python src/main_action_frame_newdata.py experiment_name=main_action_frame_newdata

# action frame xl pretrained
python src/main_action_frame_xl_pretrained_newdata.py experiment_name=main_action_frame_xl_pretrained_newdata

# action frame xl pretrained short
python src/main_action_frame_xl_pretrained_short_newdata.py experiment_name=main_action_frame_xl_pretrained_short_newdata

# Submitted batch job 54685519
# Submitted batch job 54685520
# Submitted batch job 54685521