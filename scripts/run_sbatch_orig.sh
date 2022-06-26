#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=jag-standard
#SBATCH --mem=16G
#SBATCH --exclude=jagupard[15,18]
# #SBATCH --exclude=jagupard[14,17,18,20]

# Print execute commands in the log.
set -x

# source scripts/copy_imagenet_local.sh

conda deactivate
home_dir="/sailhome/msun415"
virtual_env="/sailhome/msun415/continual_env_5-10"
project_dir="/sailhome/msun415/UCL_michael"

cd ${home_dir}
source ${virtual_env}/bin/activate

cd ${project_dir}

eval $1