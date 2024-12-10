# python make_sbatch.py --time 48 --bash_files bash_commands/1210_0_train_vda.sh

# action frame
python src/main_action_frame.py experiment_name=vda_action_frame

# action noframe
python src/main_action_noframe.py experiment_name=vda_action_noframe

# noaction frame
python src/main_noaction_frame.py experiment_name=vda_noaction_frame

# noaction noframe
python src/main_noaction_noframe.py experiment_name=vda_noaction_noframe

# Submitted batch job 54640447
# Submitted batch job 54640448
# Submitted batch job 54640449
# Submitted batch job 54640450