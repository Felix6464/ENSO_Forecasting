#!/bin/bash
#SBATCH --job-name=ddp_test
#SBATCH --nodes=1                                   # Number of nodes
#SBATCH --ntasks=2                                  # Number of tasks (processes)
#SBATCH --gres=gpu:2                                # Number of GPUs
#SBATCH --partition=2080-galvani                    # Partition to submit to gpu-2080ti-preemptable
#SBATCH --time=0-00:10:00           	            # Runtime in D-HH:MM
#SBATCH --output=logs/%j.out                        # File to which STDOUT will be written
#SBATCH --mail-type=FAIL                             # Type of email notification- BEGIN,END,FAIL,ALL 
#SBATCH --mail-user=felix.boette@student.uni-tuebingen.de  # Email to which notifications will be sent



# Activate the conda environment
source ~/.bashrc
conda activate deeps2aEnv 

set -o errexit

master_port=$(shuf -i 10000-65535 -n 1)
export MASTER_PORT=$master_port
echo "MASTER_PORT="$MASTER_PORT

#srun python ../basemodel/train_basemodel_lens.py
#srun python ../basemodel/train_basemodel_swinlstm_lens.py
srun python ../cera20c/train_basemodel_swinlstm_cera20c.py
