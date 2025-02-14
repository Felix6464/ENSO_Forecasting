#!/bin/bash

# #SBATCH --partition=cpu-long         				# Partition to submit to

#SBATCH --partition=cpu-short         				# Partition to submit to
#SBATCH --time=00:30:00            					# Runtime in D-HH:MM
#SBATCH --ntasks=1                					# Number of tasks (set for MPI, for OpenMP to 1)
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j.out  				# File to which STDOUT will be written
#SBATCH --error=logs/%j.err   				# File to which STDERR will be written

echo "RUN Script"
python ../testing/concat_taux.py \
