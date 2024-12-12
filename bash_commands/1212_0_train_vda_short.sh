# python make_sbatch.py --time 48 --gb 128 --bash_files bash_commands/1212_0_train_vda_short.sh
# extra file because it has to be 128gb

# action frame short
python src/main_action_frame_short.py experiment_name=main_action_frame_short

# Submitted batch job 54685515