# python make_sbatch.py --time 48 --bash_files bash_commands/1211_0_train_vda_rerun.sh

# action frame
python src/main_action_frame.py experiment_name=vda_action_frame

# noaction frame
python src/main_noaction_frame.py experiment_name=vda_noaction_frame

# Submitted batch job 54650378
# Submitted batch job 54650381