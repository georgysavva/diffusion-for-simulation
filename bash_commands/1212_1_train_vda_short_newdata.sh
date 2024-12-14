# python make_sbatch.py --time 48 --gb 128 --bash_files bash_commands/1212_1_train_vda_short_newdata.sh
# extra file because it has to be 128gb

# action frame short
python src/main_action_frame_short_newdata.py experiment_name=main_action_frame_short_newdata

# done care about short anymore, killed
# Submitted batch job 54685522