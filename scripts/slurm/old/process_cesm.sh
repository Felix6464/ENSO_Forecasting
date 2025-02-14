#!/bin/bash

# #SBATCH --partition=cpu-long         				# Partition to submit to

#SBATCH --partition=cpu-short         				# Partition to submit to
#SBATCH --time=05:00:00            					# Runtime in D-HH:MM
#SBATCH --ntasks=1                					# Number of tasks (set for MPI, for OpenMP to 1)
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j.out  				# File to which STDOUT will be written
#SBATCH --error=logs/%j.err   				# File to which STDERR will be written

echo "RUN Script"
python ../data_processing/process_CESM2_picontrol.py \
    -path "/mnt/qb/goswami/data/cmip6_lens/month/piControl/CESM2/ssh/" \
    -prefix b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h. \
    -outpath "./../../data/processed_data/CESM2/piControl" \
    -var SSH -cpus 8 
