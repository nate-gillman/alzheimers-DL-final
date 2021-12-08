#!/bin/bash

#SBATCH -J test_the_RNN
#SBATCH --array=0-1
#SBATCH -t 24:00:00
#SBATCH --mem=15G

#SBATCH -e experiment_output/experiment%a.err
#SBATCH -o experiment_output/experiment%a.out

# Load anaconda module
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh

# Activate virtual environment
conda activate /users/ngillman/.conda/envs/tf_env_for_DL_final

# Train and test the rnn
python rnn.py $SLURM_ARRAY_TASK_ID