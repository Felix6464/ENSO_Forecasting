import argparse
import os
import re
PATH = os.path.dirname(os.path.abspath(__file__))

def run_sbatch(command, name, partition='cpu-galvani', dry=False):
    cmd = f'sbatch -p {partition} --job-name {name} {PATH}/run.sbatch {command}'
    print(cmd)
    if not dry:
        os.system(cmd)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--partition', default='cpu-galvani', help='Partition to run on')
    parser.add_argument('-d', '--dry-run', action='store_true', help='Dry run')
    parser.add_argument('-r', '--repository', default="/mnt/qb/goswami/data/cmip6_lens/month/piControl/CESM2/taux/",help='Path to the repository containing the files')
    args = parser.parse_args()

    # Iterate over each file in the specified repository
    for filename in os.listdir(args.repository):
        print("Processing file:", filename)
        
        # Use regex to find the last number before the .nc extension
        match = re.search(r'\.(\d{6}-\d{6})\.nc$', filename)

        if match:
            # Extract the last number (year range) from the matched group
            year_range = match.group(1)
            print(f"Extracted year range: {year_range}")

            filename = filename.rsplit('.nc', 1)[0]


            # Create the name for the job
            name = f'preprocess_taux_{filename}'
            
            # Build the command dynamically
            command = f'python {PATH}/process_CESM2_picontrol_taux.py'
            command += f' -path {args.repository}'
            command += f' -prefix {filename}'  # Use the filename as the prefix
            command += f' -name CMIP6-piControl.TAUX.{year_range}'
            command += f' -var TAUX'
            command += f' -cpus 8'
            command += f' -outpath /mnt/qb/datasets/STAGING/goswami/CMIP6_LENS/CESM2/piControl/taux_1_2_grid/'

            # Execute the command
            print("Executing script")
            run_sbatch(command, name, args.partition, args.dry_run)
        else:
            print("No match found for the year range in the filename.")


if __name__ == '__main__':
    main()