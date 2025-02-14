import argparse
import os
import re
PATH = os.path.dirname(os.path.abspath(__file__))

def run_sbatch(command, name, partition='cpu-galvani', dry=False):
    cmd = f'sbatch -p {partition} --job-name {name} {PATH}/run.sbatch {command}'
    print(cmd)
    if not dry:
        os.system(cmd)
    

# Define your script's main function
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--partition', default='cpu-galvani', help='Partition to run on')
    parser.add_argument('-d', '--dry-run', action='store_true', help='Dry run')
    parser.add_argument('-r', '--repository', default="/mnt/qb/goswami/data/oras5/zonal_wind_stress/", help='Path to the repository containing the files')
    args = parser.parse_args()


    # Loop over levels
    for level in [0]:

        # Create the name for the job
        name = f'preprocess_temp_ocean_level_{level}'
        
        # Build the command dynamically
        command = f'python {PATH}/process_ORAS5_tauxy.py'
        command += f' -path {args.repository}'
        command += f' -var sozotaux'
        command += f' -cpus 8'
        command += f' -outpath /home/goswami/gkd235/deeps2a-enso/data/processed_data/enso_data_pacific/oras5/zonal_wind_stress/1_1_grid/'
        command += f' -level {level}'

        # Execute the command
        print("Executing script")
        run_sbatch(command, name, args.partition, args.dry_run)


if __name__ == '__main__':
    main()