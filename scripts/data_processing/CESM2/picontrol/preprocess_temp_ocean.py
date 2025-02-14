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
    parser.add_argument('-r', '--repository', default="/mnt/qb/goswami/data/cmip6_lens/month/piControl/CESM2/temp_ocn/", help='Path to the repository containing the files')
    args = parser.parse_args()



    # Loop over levels
    for level in [0, 1, 3, 5, 8, 11, 14]:
        # Create the name for the job
        name = f'preprocess_temp_ocean_level_{level}'
        
        # Build the command dynamically
        command = f'python {PATH}/process_CESM2_picontrol_temp_ocean.py'
        command += f' -path {args.repository}'
        command += f' -prefix _'  # Use the filename as the prefix
        command += f' -name CMIP6-piControl.TEMP.lower_levels.'
        command += f' -var TEMP'
        command += f' -cpus 8'
        command += f' -outpath /mnt/qb/datasets/STAGING/goswami/CMIP6_LENS/CESM2/piControl/temp_ocean/'
        command += f' -level {level}'

        # Execute the command
        print("Executing script")
        run_sbatch(command, name, args.partition, args.dry_run)

if __name__ == '__main__':
    main()