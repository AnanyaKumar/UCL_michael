#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --exclude=jagupard[10-25]

# Print execute commands in the log.
set -x

if [ $USER == "msun415" ]; then
	# source scripts/copy_imagenet_local.sh
    conda deactivate
    home_dir="/sailhome/msun415"
    virtual_env="/sailhome/msun415/continual_env_5-10"
    project_dir="/sailhome/msun415/UCL_michael"
    cd ${home_dir}
    source ${virtual_env}/bin/activate
    cd ${project_dir}
    eval $1

elif [ $USER == "ananya "] ; then
    conda_env=`whoami`-ue
    # source scripts/copy_imagenet_local.sh
    source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    echo $conda_env
    conda activate $conda_env
    cd $PWD
    eval $1
fi





