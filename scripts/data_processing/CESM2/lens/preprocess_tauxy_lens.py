import argparse
import os
import re
import shutil
#from s2aenso.utils.utilities import *

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

def delete_files_in_directory(dst_dir):
    # Check if the destination directory is not empty, and delete all files if necessary
    if os.listdir(dst_dir):  # Check if the directory is not empty
        print(f"Target directory {dst_dir} is not empty. Deleting all files.")
        for file_name in os.listdir(dst_dir):
            #match = re.search(r'(LE2-\d{4})\.(\d{3})', file_name)
            #ensemble_part = match.group(1) + "." + match.group(2)  # This extracts "LE2-1001.001"
            #if ensemble_part == "LE2-1301.010":
            #    print("Reached the last ensemble, stopping.")
            #    break
            file_path = os.path.join(dst_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
    

# Define your script's main function
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--partition', default='cpu-galvani', help='Partition to run on')
    parser.add_argument('-d', '--dry-run', action='store_true', help='Dry run')
    parser.add_argument('-r', '--repository', default="/mnt/qb/goswami/data/cmip6_lens/month/historical/CESM2/tauy/", help='Path to the repository containing the files')
    parser.add_argument('-o', '--output_path', default="/mnt/qb/datasets/STAGING/goswami/CMIP6_LENS/CESM2/historical_levels/tauy/1_1_grid/", help='Path to the output directory where ensemble files will be copied')

    args = parser.parse_args()

    '''
    # Dictionary to store files per ensemble
    ensemble_files = {}

    #delete_files_in_directory(args.repository)


    # Iterate over each file in the specified repository
    for filename in os.listdir(args.repository):
        # Use regex to match the part like "LE2-1001.001" in the filename
        match = re.search(r'(LE2-\d{4})\.(\d{3})', filename)

        if match:
            # Extract the ensemble part (e.g., LE2-1001)
            ensemble_part = match.group(1) + "." + match.group(2)  # This extracts "LE2-1001.001"

            # Only process files with ensemble numbers up to and including 010
            #if int(match.group(1).split("-")[-1]) > 1300:
            #    print(f"Skipping {filename} since ensemble number is greater than 1200.")
            #    break

            # If the ensemble_part is not in the dictionary, add it
            if ensemble_part not in ensemble_files:
                ensemble_files[ensemble_part] = []

            # Append the filename to the respective ensemble group
            ensemble_files[ensemble_part].append(filename)
        else:
            print(f"No match found for the ensemble part in the filename: {filename}")

    # Now, for each ensemble, copy files to respective directories and run the batch script
    for ensemble, files in ensemble_files.items():
        print(f"Processing ensemble {ensemble} with {len(files)} files.")

        # Create a directory for the ensemble
        ensemble_outputdir = os.path.join(args.output_path, f'ensemble_{ensemble.split("-")[-1].replace('.', '_')}')
        ensemble_datadir = os.path.join("/mnt/qb/goswami/data/cmip6_lens/month/historical/CESM2/tauy/", f'ensemble_{ensemble.split("-")[-1].replace('.', '_')}')
        delete_files_in_directory(ensemble_datadir)
        #create_directory(ensemble_datadir)
        create_directory(ensemble_outputdir)

        # Copy files to the respective ensemble directory
        for file in files:
            src_file = os.path.join(args.repository, file)
            print(f"Copying {file} to {ensemble_datadir}")
            copy_file_to_ensemble_dir(src_file, ensemble_datadir)'''


    name = f'preprocess_tauxy'

    # Build the command dynamically
    command = f'python {PATH}/process_CESM2-LENS.py'
    command += f' -path {args.repository}'  # Use the ensemble directory as the input path
    command += f' -prefix _'  # Use the ensemble as the prefix
    command += f' -name TAUY'
    command += f' -var TAUY'
    command += f' -cpus 8'
    command += f' -outpath {args.output_path}'
    command += f' -level {0}'
    command += f' -chunks {18}'

    # Execute the command for the ensemble directory
    print(f"Executing batch script")
    run_sbatch(command, name, args.partition, args.dry_run)

if __name__ == '__main__':
    main()