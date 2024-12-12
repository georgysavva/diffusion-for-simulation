# python make_sbatch.py --time 2 --gb 32 --bash_files bash_commands/1211_2_generate_data_new.sh

# mod 0
python scripts/preprocess_data_with_vae.py --dataset_type train --div 10 --mod 0

# mod 1
python scripts/preprocess_data_with_vae.py --dataset_type train --div 10 --mod 1

# mod 2
python scripts/preprocess_data_with_vae.py --dataset_type train --div 10 --mod 2

# mod 3
python scripts/preprocess_data_with_vae.py --dataset_type train --div 10 --mod 3

# mod 4
python scripts/preprocess_data_with_vae.py --dataset_type train --div 10 --mod 4

# mod 5
python scripts/preprocess_data_with_vae.py --dataset_type train --div 10 --mod 5

# mod 6
python scripts/preprocess_data_with_vae.py --dataset_type train --div 10 --mod 6

# mod 7
python scripts/preprocess_data_with_vae.py --dataset_type train --div 10 --mod 7

# mod 8
python scripts/preprocess_data_with_vae.py --dataset_type train --div 10 --mod 8

# mod 9
python scripts/preprocess_data_with_vae.py --dataset_type train --div 10 --mod 9

# test
python scripts/preprocess_data_with_vae.py --dataset_type test --div 1 --mod 0

# Submitted batch job 54683514
# Submitted batch job 54683515
# Submitted batch job 54683516
# Submitted batch job 54683517
# Submitted batch job 54683518
# Submitted batch job 54683519
# Submitted batch job 54683520
# Submitted batch job 54683521
# Submitted batch job 54683522
# Submitted batch job 54683523
# Submitted batch job 54683524