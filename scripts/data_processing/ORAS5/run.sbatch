#!/bin/bash
#mkdir -p logs # Creates a directory to store logs

#SBATCH --partition=cpu-galvani
#SBATCH --job-name=temp_ocean_cesm_preprocessing
#SBATCH --time=04:00:00            					# Runtime in D-HH:MM
#SBATCH --ntasks=1                					# Number of tasks (set for MPI, for OpenMP to 1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=800GB
#SBATCH --output=logs/%j.out                        # File to which STDOUT will be written


source ~/.bashrc
source $PREAMBLE
conda activate deeps2aEnv

unset SLURM_CPU_BIND

srun "$@"