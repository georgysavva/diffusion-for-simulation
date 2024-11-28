# python make_sbatch.py --time 48 --bash_files bash_commands/1127_0_generate_data.sh

# generate 1 episode
python scripts/collect_doom_data.py --save_path vae_data --num_episodes 5000 --observations_only --skip_frames 5

# Submitted batch job 54073377