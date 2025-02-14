#!/bin/bash
#SBATCH --job-name=vit_cmip6
#SBATCH --nodes=1                                   # Number of nodes
#SBATCH --ntasks=1                                  # Number of tasks (processes)
#SBATCH --gres=gpu:1                                # Number of GPUs
#SBATCH --partition=a100-galvani                    # Partition to submit to gpu-2080ti-preemptable
#SBATCH --mem=400GB
#SBATCH --time=0-50:00:00           	            # Runtime in D-HH:MM
#SBATCH --output=logs/%j.out                        # File to which STDOUT will be written
#SBATCH --mail-type=END,FAIL                             # Type of email notification- BEGIN,END,FAIL,ALL 
#SBATCH --mail-user=felix.boette@student.uni-tuebingen.de  # Email to which notifications will be sent



# Activate the conda environment
source ~/.bashrc
conda activate deeps2aEnv 

set -o errexit

# Uncomment this if using SLURM and need to determine the master address
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR

# Randomly select a master port
master_port=$(shuf -i 10000-65535 -n 1)
export MASTER_PORT=$master_port
echo "MASTER_PORT="$MASTER_PORT

# Run the Python script and pass any additional parameters to it
srun python ../zhou_data/single_gpu_vit_cmip6.py "$@"
