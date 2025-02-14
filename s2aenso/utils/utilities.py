import pickle
import torch
import io, os
import numpy as np
import random



def count_parameters(model, min_param_size = 512):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_params = parameter.numel()
        total_params += param_params
        if param_params > min_param_size:
            print(f"{name}: {param_params}")
    print(f"Total Parameters: {total_params}")


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
        

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


def set_seed(seed_value):
    # Set the seed for generating random numbers in PyTorch
    torch.manual_seed(seed_value)
    
    # Set the seed for CUDA (if using GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # If using multiple GPUs
    
    # Ensure that the results are reproducible even for certain layers, such as dropout layers
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the seed for other libraries like numpy and random
    np.random.seed(seed_value)
    random.seed(seed_value)


def calculate_mean_std(loss_dict, metric):
    all_values = [loss_dict[subsample][metric] for subsample in loss_dict if 'total' not in subsample]
    mean_values = np.mean(all_values, axis=0)
    std_values = np.std(all_values, axis=0)
    return mean_values, std_values