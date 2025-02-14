#!/bin/bash
#mkdir -p logs # Creates a directory to store logs

##SBATCH --partition=cpu-galvani
#SBATCH --partition=2080-galvani
#SBATCH --mem=40GB
#SBATCH --job-name=process_samples
#SBATCH --time=01:30:00            					# Runtime in D-HH:MM
##SBATCH --ntasks=1                					# Number of tasks (set for MPI, for OpenMP to 1)
##SBATCH --cpus-per-task=1
#SBATCH --output=logs/%j.out                        # File to which STDOUT will be written


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

srun python ../picontrol/evaluation/evaluation_plots/process_samples.py