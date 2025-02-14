#!/bin/bash
#SBATCH --job-name=base_lens
#SBATCH --nodes=1                                   # Number of nodes
#SBATCH --ntasks=4                                 # Number of tasks (processes)
#SBATCH --gres=gpu:4                                # Number of GPUs
#SBATCH --partition=a100-galvani                    # Partition to submit to gpu-2080ti-preemptable
#SBATCH --time=0-05:00:00                           # Runtime in D-HH:MM
#SBATCH --output=logs/%j.out                        # File to which STDOUT will be written
#SBATCH --mail-type=END,FAIL                             # Type of email notification- BEGIN,END,FAIL,ALL 
#SBATCH --mail-user=felix.boette@student.uni-tuebingen.de  # Email to which notifications will be sent



# Activate the conda environment
source ~/.bashrc
conda activate deeps2aEnv 

set -o errexit

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR

master_port=$(shuf -i 10000-65535 -n 1)
export MASTER_PORT=$master_port
echo "MASTER_PORT="$MASTER_PORT

srun python ../picontrol/distributed_swinlstm_picontrol.py
