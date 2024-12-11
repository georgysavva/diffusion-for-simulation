import argparse
import os
import glob

template = """#!/bin/bash

#SBATCH --constraint='a100|h100'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$NCPU
#SBATCH --time=$TIME:00:00
#SBATCH --mem=$MEMGB
$GPULINE
#SBATCH --job-name=$JOBNAME
#SBATCH --output=slurm_outs/%j.out

module purge

singularity exec --nv \\
    --overlay /scratch/yl11330/my_env/overlay-50G-10M-pytorch.ext3:ro \\
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \\
    /bin/bash -c "source /ext3/env.sh; cd /scratch/yl11330/diffusion-for-simulation; conda activate ./penv; \\
        $CMD"
"""

parser = argparse.ArgumentParser()
parser.add_argument('--bash_files', type=str, nargs='+', help='bash file of commands', required=True)
parser.add_argument('--gb', type=int, help='bash file of commands', default=64)
parser.add_argument('--ncpu', type=int, help='bash file of commands', default=8)
parser.add_argument('--time', type=str, help='bash file of commands', required=True)
parser.add_argument('--sbatch_dir', type=str, help='bash file of commands', default='/scratch/yl11330/diffusion-for-simulation/sbatch_files')
args = parser.parse_args()

# remove existing sbatch files
for sbatch_file in glob.glob(os.path.join(args.sbatch_dir, '*')):
    os.remove(sbatch_file)

template = template.replace('$NCPU', str(args.ncpu))
template = template.replace('$TIME', str(args.time))
template = template.replace('$MEM', str(args.gb))

# todo: support multi-gpu
gpu_line = '#SBATCH --gres=gpu:1'
template = template.replace('$GPULINE', gpu_line)

# get job clusters from bash files
model_dirs_dict = {}
for bash_file in args.bash_files:
    # filter lines
    orig_lines = open(bash_file, 'r').readlines()
    orig_lines = [l.strip() for l in orig_lines if l.strip()]
    orig_lines = [l for l in orig_lines if 'Submitted batch job' not in l]
    # collapse "\\"
    lines = []
    add_to_previous_line = False
    for l in orig_lines:
        if l.startswith('#'):
            lines.append(l)
            add_to_previous_line = False
        elif l.endswith('\\'):
            l = l[:-1]
            if add_to_previous_line:
                lines[-1] += l
            else:
                lines.append(l)
            add_to_previous_line = True
        else:
            if add_to_previous_line:
                lines[-1] += l
            else:
                lines.append(l)
            add_to_previous_line = False
    # just for assertions
    job_lines = [l for l in lines if not l.startswith('#')]
    assert len(job_lines) == len(set(job_lines)), 'duplicate jobs'
    if '--tag' in job_lines[0]:
        tags = [l.split()[l.split().index('--tag') + 1] for l in job_lines]
        assert len(tags) == len(set(tags)), 'duplicate tags'
    # collect clusters
    job_cluster_names = []
    job_clusters = []
    for i, l in enumerate(lines):
        if l.startswith('#'):
            name = l[1:].strip()
            job_cluster_names.append(name)
            job_clusters.append([])
        else:
            job_clusters[-1].append(l)
        i += 1
    assert len(job_cluster_names) == len(job_clusters)
    # filter out empty clusters (random comments)
    for cluster_name, job_cluster in zip(job_cluster_names, job_clusters):
        if len(job_cluster) >= 1:
            model_dirs_dict[cluster_name] = job_cluster


for cluster_name, job_cluster in model_dirs_dict.items():
    for job_i, cmd in enumerate(job_cluster):
        # create sbatch content
        job_name = f"{cluster_name.replace(' ', '_')}_{job_i}"
        sbatch_content = template.replace('$JOBNAME', job_name)
        sbatch_content = sbatch_content.replace('$CMD', cmd)
        assert '$' not in sbatch_content
        # save sbatch content
        sbatch_path = os.path.join(args.sbatch_dir, f'{job_name}.sbatch')
        print(sbatch_path)
        with open(sbatch_path, 'w') as f:
            f.write(sbatch_content)
