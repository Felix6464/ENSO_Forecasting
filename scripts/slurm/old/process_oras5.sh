#!/bin/bash

#SBATCH --partition=cpu-galvani        				# Partition to submit to
#SBATCH --time=05:00:00            					# Runtime in D-HH:MM
#SBATCH --ntasks=1                					# Number of tasks (set for MPI, for OpenMP to 1)
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j.out  				# File to which STDOUT will be written
#SBATCH --error=logs/%j.err   				# File to which STDERR will be written
#SBATCH --mail-type=FAIL                             # Type of email notification- BEGIN,END,FAIL,ALL 
#SBATCH --mail-user=felix.boette@student.uni-tuebingen.de  # Email to which notifications will be sent

# Activate the conda environment
source ~/.bashrc
source $PREAMBLE
conda activate deeps2aEnv 

echo "RUN Script"
start_time=$(date +%s)

# SST
# python ../data_processing/process_ORAS5.py \
#     -path "/mnt/qb/datasets/STAGING/goswami/oras5/sst/" \
#     -prefix "sea_surface_temperature_oras5_single_level_" \
#     -var 'sosstsst' -cpus 8 \
#     -outpath "./../data/processed_data/oras5/"

# SSH
python ../data_processing/process_ORAS5.py \
    -path "/mnt/qb/datasets/STAGING/goswami/oras5/ssh/" \
    -prefix "sea_surface_height_oras5_single_level_" \
    -var 'sossheig' -cpus 8 \
    -outpath "./../../data/processed_data/oras5/"


# Get runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))

printf "Runtime: %02d:%02d:%02d\n" $((runtime/3600)) $(((runtime%3600)/60)) $((runtime%60))