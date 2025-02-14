import argparse
import os
import re
import shutil


PATH = os.path.dirname(os.path.abspath(__file__))

def run_sbatch(command, name, partition='cpu-galvani', dry=False):
    cmd = f'sbatch -p {partition} --job-name {name} {PATH}/run.sbatch {command}'
    print(cmd)
    if not dry:
        os.system(cmd)

def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def copy_file_to_ensemble_dir(src_path, dst_dir):
    """Copy a file to the ensemble directory if it doesn't exist there, and then delete the source file."""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Check if the file already exists in the destination
    dst_file = os.path.join(dst_dir, os.path.basename(src_path))
    if not os.path.exists(dst_file):
        # Copy the file
        shutil.copy(src_path, dst_file)
        print(f"Copied {src_path} to {dst_dir}")
        
        # Delete the file from the source directory after copying
        os.remove(src_path)
        print(f"Deleted {src_path} after copying.")
    else:
        print(f"File {os.path.basename(src_path)} already exists in {dst_dir}, skipping.")
    

# Define your script's main function
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--partition', default='cpu-galvani', help='Partition to run on')
    parser.add_argument('-d', '--dry-run', action='store_true', help='Dry run')
    parser.add_argument('-r', '--repository', default="/mnt/qb/goswami/data/cmip6_lens/month/historical/CESM2/temp_levels/", help='Path to the repository containing the files')
    parser.add_argument('-o', '--output_path', default="/mnt/qb/datasets/STAGING/goswami/CMIP6_LENS/CESM2/historical_levels/temp_ocean/lower_levels/1_2_grid/final_ensembles/", help='Path to the output directory where ensemble files will be copied')

    args = parser.parse_args()

    #for level in [0, 4, 8, 12, 16, 20, 24]:
    #for level in [0, 1, 3, 5, 8, 11, 14]:
    for level in [8]:
      
        name = f'preprocess_temp_ocean_level_{level}'

        # Build the command dynamically
        command = f'python {PATH}/process_CESM2-LENS.py'
        command += f' -path {args.repository}'  # Use the ensemble directory as the input path
        command += f' -prefix _'  # Use the ensemble as the prefix
        command += f' -name TEMP'
        command += f' -var TEMP'
        command += f' -cpus 8'
        command += f' -outpath {args.output_path}'
        command += f' -level {level}'
        command += f' -chunks {18}'

        # Execute the command for the ensemble directory
        print(f"Executing batch script")
        run_sbatch(command, name, args.partition, args.dry_run)
        

if __name__ == '__main__':
    main()