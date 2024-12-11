# python make_sbatch.py --time 48 --bash_files bash_commands/1211_0_train_vda_xl.sh

# action frame xl short
python src/main_action_frame_xl_short.py experiment_name=main_action_frame_xl_short

# action frame xl pretrained short
python src/main_action_frame_xl_pretrained_short.py experiment_name=main_action_frame_xl_pretrained_short

# action frame xl
python src/main_action_frame_xl.py experiment_name=main_action_frame_xl

# action frame xl pretrained
python src/main_action_frame_xl_pretrained.py experiment_name=main_action_frame_xl_pretrained

# noaction frame xl short
python src/main_noaction_frame_xl_short.py experiment_name=main_noaction_frame_xl_short

# noaction frame xl pretrained short
python src/main_noaction_frame_xl_pretrained_short.py experiment_name=main_noaction_frame_xl_pretrained_short

# noaction frame xl
python src/main_noaction_frame_xl.py experiment_name=main_noaction_frame_xl

# noaction frame xl pretrained
python src/main_noaction_frame_xl_pretrained.py experiment_name=main_noaction_frame_xl_pretrained
