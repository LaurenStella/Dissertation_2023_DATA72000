#!/bin/bash

# Generic options:

#SBATCH --account=bdman20  # Run job under project <project>
#SBATCH --time=05:00:0         # Run for a max of 1 hour

# Node resources:
# (choose between 1-4 gpus per node)

#SBATCH --partition=gpu    # Choose either "gpu" or "infer" node type
#SBATCH --nodes=1          # Resources from a single node
#SBATCH --gres=gpu:2       # One GPU per node (plus 25% of node CPU and RAM per GPU)

# Run commands:
source /nobackup/projects/bdman20
#source activate flaml
source activate flamlclone
python3 'Autoreg-2_linear.py' # Display available gpu resources

# Place other commands here
conda deactivate
echo "end of job"
