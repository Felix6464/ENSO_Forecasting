''' Distributed data loading and normalization for CESM-LENS data. 

@Author  :   Sebastian Hofmann, Jannik Thuemmel, Jakob Schlör 
@Time    :   2024/02/02 14:21:48
@Contact :   jakob.schloer@uni-tuebingen.de
'''


from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
import einops



import s2aenso.utils.normalization as normalization

# removed dmlcloud dependency
# =============================================================================
def print_worker(msg, barrier=True, flush=True):
    if barrier:
        dist.barrier()
    print(f'Worker {dist.get_rank()} ({dist.get_group_rank()}.{dist.get_process_group_ranks()}): {msg}', flush=flush)
    if barrier:
        dist.barrier()


def shard_indices(n, rank, size, shuffle=True, drop_remainder=False, seed=0):
    indices = np.arange(n)

    if shuffle:
        np.random.Generator(np.random.MT19937(seed)).shuffle(indices)

    if drop_remainder:
        indices = indices[: n - n % size]

    return indices[rank::size]


def load_netcdf_dataset(path, variables=None, dtype=np.float32):
    """
    Loads all .nc (netcdf) files, and filters to the given variables.
    Returns a list of xarray.DataArray objects.
    If variables is None or empty, all variables are returned.
    """
    files = list(sorted(Path(path).glob('*.nc')))
    if len(files) == 0:
        raise ValueError(f'No netcdf files found in {path}')

    data_arrays = []
    for file in files:
        da = xr.open_dataarray(file)
        if variables and da.name not in variables:
            continue
        if 'time' not in da.sizes:
            raise ValueError('Dataset must have a time dimension')
        if 'lat' not in da.sizes:
            raise ValueError('Dataset must have a lat dimension')
        if 'lon' not in da.sizes:
            raise ValueError('Dataset must have a lon dimension')
        if dtype is not None:
            da = da.astype(dtype)
        
        # Slice the DataArray on the time dimension
        data_arrays.append(da)



    if variables and (len(data_arrays) != len(set(variables))):
        missing = set(variables) - {da.name for da in data_arrays}
        raise ValueError(f'Not all variables found. Missing: {missing}')

    return data_arrays

# ======================================================================================
# Dataset classes for CESM2-LENS data 
# ======================================================================================
class NetcdfGeoDataset(torch.utils.data.Dataset):
    """
    Temporal-spatial dataset stored as a collection of netcdf files.
    Each netcdf file must contain a single variable with dimensions (time, lat, lon).
    Resulting dataset has shape (num_samples, num_variables, sequence_length, lat, lon).
    """

    def __init__(self, path: str, vars: list, sequence_length:int, 
                 normalizer=None, transform=None):
        self.path = Path(path)
        self.vars = vars
        self.sequence_length = sequence_length
        self.normalizer = normalizer
        self.transform = transform

        dataset = self._load_dataset()
        self.dataset_xr = self._load_dataset()
        #print("Self.vars : ", self.vars)
        self._channel_indices = [self.variable_mapping[var] for var in self.vars]
        #print("Variable mapping : ", self.variable_mapping)
        #print("Channel indices : ", self._channel_indices)
        
        self._to_array(dataset)

    def _load_dataset(self):
        data_arrays = load_netcdf_dataset(self.path, self.vars)
        if len(data_arrays) == 0:
            raise ValueError(f'No variables found in {self.path}')

        # Extract land-sea mask
        lsm = data_arrays[0].isel(time=0).isnull().astype(np.float32)
        lsm.name = 'lsm'
        del lsm['month']
        del lsm['time']
        if 'lsm' in self.vars:
            data_arrays.append(lsm)
        self.lsm = lsm.data

        self.variable_mapping = {da.name: i for i, da in enumerate(data_arrays)}
        dataset = xr.merge(data_arrays, join='inner')
        return dataset

    def _to_array(self, dataset):
        np_arr = dataset.to_array(dim='variable').values  # (variable, time, lat, lon)
        np.nan_to_num(np_arr, copy=False, nan=0.0)
        self.data = torch.from_numpy(np_arr).float().share_memory_()
        self.coords = {coord: dataset.coords[coord].values for coord in dataset.coords}
        print("Data shape: ", self.data.shape)

        if self.normalizer is not None:
            print("Using given normalizer")
            self.data = self.data[self._channel_indices, :, :, :]
            self.data = self.normalizer(self.data.unsqueeze(0)).squeeze(0)  # (variable, time, lat, lon)
            self.data.share_memory_()
        else:
            print(f"Creating normalizer for {self.vars}")
            self.data = self.data[self._channel_indices, :, :, :]
            self.normalizer = normalization.Normalizer(variables=self.vars)
            _ = self.normalizer.fit2xr(self.dataset_xr)
            self.data = self.normalizer(self.data.unsqueeze(0)).squeeze(0)  # (variable, time, lat, lon)
            self.data.share_memory_()

    def __len__(self):
        return self.data.shape[1] - self.sequence_length + 1

    def __getitem__(self, idx):
        x = self.data[:, idx : idx + self.sequence_length]
        print("X shape: ", x.shape)
        month = self.coords['month'][idx : idx + self.sequence_length]
        month = month - 1  # 1-indexed to 0-indexed
        sample = {
            'x': x,
            'month': month,
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

class ZhouDataset(torch.utils.data.ConcatDataset):
    def __init__(self, path, vars, sequence_length, transform=None):
        datasets = []

        # Load the full dataset
        full_dataset = xr.open_dataset(path)
        print("Dataset_loaded: ", full_dataset)
        
        # Iterate over the n_model dimension and create sub-datasets
        for model_index in range(full_dataset.sizes['n_model']):
        #for model_index in range(2):
            model_dataset = full_dataset.isel(n_model=model_index)
            ds = NetcdfGeoDataset_Zhou(model_dataset, vars, sequence_length, transform)
            print("Dataset number {} loaded".format(model_index))
            datasets.append(ds)

        super().__init__(datasets)
        self.transform = transform

    @property
    def lsm(self):
        return self.datasets[0].lsm

def load_zhou_dataset(data_path, vars, sequence_length):
    return ZhouDataset(data_path, vars, sequence_length)

def load_zhou_godas(data_path, vars, sequence_length):
    full_dataset = xr.open_dataset(data_path)
    return NetcdfGeoDataset_Zhou(full_dataset, vars, sequence_length)

def load_zhou_dataset_val(data_path, vars, sequence_length):
    full_dataset = xr.open_dataset(data_path)
    return NetcdfGeoDataset_Zhou_Val(full_dataset, vars, sequence_length)


class NetcdfGeoDataset_Zhou_Val(torch.utils.data.Dataset):
    """
    Temporal-spatial dataset stored in a single NetCDF file.
    The dataset has dimensions (n_mon, lev, lat, lon).
    Resulting dataset has shape (num_samples, num_variables, sequence_length, lat, lon).
    """

    def __init__(self, dataset_xr: xr.Dataset, vars: list, sequence_length: int, transform=None):
        self.dataset_xr = dataset_xr
        self.vars_in = ["temperatureNor_in", "tauxNor_in", "tauyNor_in"]
        self.vars_out = ["temperatureNor_out", "tauxNor_out", "tauyNor_out"]
        self.sequence_length = sequence_length
        self.transform = transform

        # Ensure the dataset contains the required variables
        self._validate_variables()

        # Prepare the dataset as a PyTorch-compatible array
        self._to_array()

    def _validate_variables(self):
        """Ensure all specified variables exist in the dataset."""
        missing_vars = [var for var in self.vars_in if var not in self.dataset_xr.data_vars]
        if missing_vars:
            raise ValueError(f"Variables {missing_vars} not found in the dataset.")

    def _to_array(self):
        """Convert the dataset to a PyTorch-compatible array."""
        datasets_np_in = []
        datasets_np_out = []

        for var in self.vars_in:
            dataset = self.dataset_xr[[var]]
            #print("Dataset: ", dataset)

            # Convert to a NumPy array
            np_arr = dataset.to_array(dim='variable').values  # Shape: (variable, n_mon, lev, lat, lon)
            np.nan_to_num(np_arr, copy=False, nan=0.0)  # Replace NaNs with 0

            # Convert to PyTorch tensor
            self.data = torch.from_numpy(np_arr).float().share_memory_()
            if var in ["tauxNor_in", "tauyNor_in"]:
                self.data = einops.rearrange(self.data, 'C T S H W -> C S T H W')  # Rearrange for PyTorch
            else:
                self.data = einops.rearrange(self.data[0], 'T S C H W -> C S T H W')  # Rearrange for PyTorch
            datasets_np_in.append(self.data)

        # Concatenate all datasets along the first dimension (n_mon dimension)
        if datasets_np_in:
            self.data_in = torch.cat(datasets_np_in, dim=0)  # Concatenate tensors along the first dimension
        else:
            print("No datasets to concatenate. Check the input variables and dataset.")

        for var in self.vars_out:
            dataset = self.dataset_xr[[var]]
            #print("Dataset: ", dataset)

            # Convert to a NumPy array
            np_arr = dataset.to_array(dim='variable').values  # Shape: (variable, n_mon, lev, lat, lon)
            np.nan_to_num(np_arr, copy=False, nan=0.0)  # Replace NaNs with 0

            # Convert to PyTorch tensor
            self.data = torch.from_numpy(np_arr).float().share_memory_()
            if var in ["tauxNor_out", "tauyNor_out"]:
                self.data = einops.rearrange(self.data, 'C T S H W -> C S T H W')  # Rearrange for PyTorch
            else:
                self.data = einops.rearrange(self.data[0], 'T S C H W -> C S T H W')  # Rearrange for PyTorch
            datasets_np_out.append(self.data)

        # Concatenate all datasets along the first dimension (n_mon dimension)
        if datasets_np_out:
            self.data_out = torch.cat(datasets_np_out, dim=0)  # Concatenate tensors along the first dimension
        else:
            print("No datasets to concatenate. Check the input variables and dataset.")

        # Remove the first variable (sst_depth=0) from both arrays
        self.data_in = self.data_in[1:, :, :, :, :]  # Shape after slicing: [D-1, B, T, H, W]
        self.data_out = self.data_out[1:, :, :, :, :]  # Shape after slicing: [D-1, B, T, H, W]

        # Concatenate the arrays along the 2nd dimension (index 1)
        self.data = torch.cat((self.data_in, self.data_out), dim=1)
        print("Data shape: ", self.data.shape)
        #[C, L, T, H, W]

        dataset_in = self.dataset_xr[self.vars_in]
        dataset_out = self.dataset_xr[self.vars_out]

        dataset_out = dataset_out.rename_dims({"Tout": "lookback"})

        # Concatenate the two datasets along the 'lookback' dimension
        dataset = xr.concat([dataset_in, dataset_out], dim="lookback")

        # Create cyclic months for each group, ensuring the first group starts at 0 and subsequent groups shift by 1
        num_groups = dataset.sizes['group']
        lookback = dataset.sizes['lookback']

        months = np.zeros((num_groups, lookback), dtype=int)
        for g in range(num_groups):
            months[g, :] = np.arange(lookback) % 12 + g  # Shift each group
            months[g, :] = months[g, :] % 12  # Ensure cyclic behavior within [0, 11]

        # Assign 'month' as a new data variable
        dataset = dataset.assign(month=(['group', 'lookback'], months))

        print("Dataset: ", dataset)

        dataset_month = dataset[['month']]
        np_arr = dataset_month.to_array(dim='variable').values
        self.data_month = torch.from_numpy(np_arr).float().share_memory_()

        # Store coordinates for easy access
        self.coords = {coord: dataset.coords[coord].values for coord in dataset.coords}

    def __len__(self):
        return self.data.shape[2] #- self.sequence_length + 1

    def __getitem__(self, idx):
        """Retrieve a single sample from the dataset."""
        # Check if the index is valid
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for the dataset.")

        # Extract the sequence for the specified variables
        x = self.data[:, :, idx, :, :]

        # Extract the month information
        month = None
        if self.data_month is not None:
            month = self.data_month[0, idx, :]

        # Prepare the sample
        sample = {
            'x': x,
            'month': month,  # Include month in the sample
        }

        return sample

class NetcdfGeoDataset_Zhou(torch.utils.data.Dataset):
    """
    Temporal-spatial dataset stored in a single NetCDF file.
    The dataset has dimensions (n_mon, lev, lat, lon).
    Resulting dataset has shape (num_samples, num_variables, sequence_length, lat, lon).
    """

    def __init__(self, dataset_xr: xr.Dataset, vars: list, sequence_length: int, transform=None):
        self.dataset_xr = dataset_xr
        self.vars = vars
        self.sequence_length = sequence_length
        self.transform = transform

        # Ensure the dataset contains the required variables
        self._validate_variables()

        # Prepare the dataset as a PyTorch-compatible array
        self._to_array()

    def _validate_variables(self):
        """Ensure all specified variables exist in the dataset."""
        missing_vars = [var for var in self.vars if var not in self.dataset_xr.data_vars]
        if missing_vars:
            raise ValueError(f"Variables {missing_vars} not found in the dataset.")

    def _to_array(self):
        """Convert the dataset to a PyTorch-compatible array."""
        
        #print("Starting to convert the dataset to a PyTorch-compatible array.")
        datasets_np = []

        for var in self.vars:
            dataset = self.dataset_xr[[var]]
            #print("Dataset: ", dataset)

            # Convert to a NumPy array
            np_arr = dataset.to_array(dim='variable').values  # Shape: (variable, n_mon, lev, lat, lon)
            np.nan_to_num(np_arr, copy=False, nan=0.0)  # Replace NaNs with 0

            # Convert to PyTorch tensor
            self.data = torch.from_numpy(np_arr).float().share_memory_()
            if var in ["tauxNor", "tauyNor"]:
                pass
            else:
                self.data = einops.rearrange(self.data[0], 'T C H W -> C T H W')  # Rearrange for PyTorch
            datasets_np.append(self.data)

            # Concatenate all datasets along the first dimension (n_mon dimension)
        if datasets_np:
            self.data = torch.cat(datasets_np, dim=0)  # Concatenate tensors along the first dimension
            print("Data shape: ", self.data.shape)
        else:
            print("No datasets to concatenate. Check the input variables and dataset.")
            print("Data shape:", self.data.shape)

        self.data = self.data[1:, :, :, :]  # Remove the first variable sst_depth=0
        dataset = self.dataset_xr[self.vars]
        months = (np.arange(dataset.sizes['n_mon']) % 12) # Cyclic months (1 to 12)
        dataset = dataset.assign(month=('month', months))  # Add 'month' as a data variable
        # Store coordinates for easy access
        self.coords = {coord: dataset.coords[coord].values for coord in dataset.coords}

    def __len__(self):
        return self.data.shape[1] - self.sequence_length + 1

    def __getitem__(self, idx):
        """Retrieve a single sample from the dataset."""
        # Check if the index is valid
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for the dataset.")

        # Extract the sequence for the specified variables
        x = self.data[:, idx: idx + self.sequence_length, :, :]
        # Extract the month information
        month = None
        if 'month' in self.coords:
            month = self.coords['month'][idx: idx + self.sequence_length]

        # Prepare the sample
        sample = {
            'x': x,
            'month': month,  # Include month in the sample
        }

        return sample



class CesmLensDataset(torch.utils.data.ConcatDataset):
    def __init__(self, directories, vars, sequence_length, transform=None, normalizer=None):
        assert len(directories) > 0
        datasets = []
        for directory in directories:
            ds = NetcdfGeoDataset(directory, vars, sequence_length,
                                  normalizer=normalizer,
                                  transform=transform)
            datasets.append(ds)
        super().__init__(datasets)
        self.transform = transform

    @property
    def lsm(self):
        return self.datasets[0].lsm


def split_cesm_lens(root_dir, num_directories_split=80, max_split=100):
    """
    Looks for CESM Large Ensemble data in root_dir and splits it into train, val.
    Returns a 2-tuple of lists of directories.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise ValueError(f'{root_dir} does not exist')

    directories = [path for path in root_dir.iterdir() if path.is_dir()]
    if len(directories) == 0:
        raise ValueError(f'No directories found in {root_dir}')

    N = len(directories)
    print("Length directories : ", N)
    train_dirs = directories[:num_directories_split] 
    val_dirs = directories[num_directories_split:max_split]
    print("Length train dirs:", len(train_dirs))
    print("Length val dirs:", len(val_dirs))
    return train_dirs, val_dirs


def shard_cesm_lens(directories, rank, size, seed=0, shuffle=True, drop_remainder=True):
    """
    Given a list of directories, return the subset/shard of directories
    that should be used by the given rank.
    """
    indices = shard_indices(len(directories), rank, size, shuffle=shuffle, drop_remainder=drop_remainder, seed=seed)
    return [directories[i] for i in indices]


def load_cesm_lens(root_dir, vars, sequence_length, split, rank, size, transform=None, seed=0, num_directories_split=80, max_split=100, normalizer=None):
    train_dirs, val_dirs = split_cesm_lens(root_dir, num_directories_split, max_split)
    if split == 'train':
        directories = shard_cesm_lens(train_dirs, rank, size, seed=seed)
    elif split == 'val':
        directories = shard_cesm_lens(val_dirs, rank, size, seed=seed, shuffle=False, drop_remainder=True)
    else:
        raise ValueError(f'Invalid split: {split}')
    

    return CesmLensDataset(directories, vars, sequence_length, transform=transform, normalizer=normalizer)



def load_picontrol(root_dir, vars, sequence_length, split, 
                    normalizer=None, transform=None, test=False):
    """Load reanalysis dataset, either ORAS5 or CERA-20C.
    
    Args:
        root_dir (str): Folder containing netcdf files.
        vars (list): Variables.
        sequence_length (int): Sequence length.
        split (str): Split of dataset, either "train" or "val".
        lsm_path (str): Path to common land-sea mask between datasets.
        normalizer (_type_, optional): Normalizer of data. For split='val' a normalizer needs to be set. Defaults to None.
        transform (_type_, optional): Pytorch transfrom . Defaults to None.
    Returns:
        NetcdfReanalysisForecastingDataset: Dataset object.
    """
    if split == 'train':
        if test is False:
            date_range = (None, '1699-12-31')
        else: 
            date_range = (None, '0769-12-31')
    elif split == 'val':
        if test is False:
            date_range = ('1700-01-01', None)
        else: 
            date_range = ('0770-01-01', None)
        if normalizer is None:
            #raise ValueError('normalizer must be provided for val split')
            print("No Normalizer given")
    elif split == 'all':
        date_range = (None, None)
    else:
        raise ValueError(f'Invalid split: {split}')

    dataset = NetcdfReanalysisDataset(root_dir, vars, sequence_length,
                                      date_range, normalizer, transform)

    return dataset


# ======================================================================================
# Dataset classes for reanalysis data 
# ======================================================================================
class NetcdfReanalysisDataset(torch.utils.data.Dataset):
    """Dataset class for spatial-temporal forecasting of reanalysis data. 

    Args:
        path (str): Folder containing netcdf files. 
        vars (list): Input variables.
        sequence_lenght (int): Input sequence length.
        normalizer (_type_, optional): Normalizer of data. If None create new normalizer. Defaults to None.
        transform (_type_, optional): Pytorch transfrom . Defaults to None.
    """
    def __init__(self, path: str, vars: list,  
                 sequence_length: int, 
                 date_range: tuple=(None, None),
                 normalizer=None, transform=None):
        super().__init__()
        self.path = path
        self.vars = vars
        self.sequence_length = sequence_length
        self.transform = transform
        self.date_range = date_range

        # Load dataset
        self.dataset = self._load_dataset()

        self.lsm = self._get_lsm()

        # Create normalizer
        if normalizer is None:
            print(f"Creating new normalizer for {self.vars}")
            self.normalizer = normalization.Normalizer(variables=self.vars)
            _ = self.normalizer.fit2xr(self.dataset)
        else:
            print("Using given normalizer")
            self.normalizer = normalizer 

        # Create data array from dataset
        self._to_array(self.dataset)

        self.variable_mapping = {var: i for i, var in enumerate(self.vars)}
        self._channel_indices = [self.variable_mapping[var] for var in self.vars]

    def _load_dataset(self):
        """Load dataset from netcdf files and create land-sea mask. """
        data_arrays = load_netcdf_dataset(self.path, self.vars)
        if len(data_arrays) == 0:
            raise ValueError(f'No variables found in {self.path}')

        # Print and update the date range format for each DataArray before slicing
        for da in data_arrays:
            # Print the updated date range
            start_date = da['time'].min()
            end_date = da['time'].max()
            #print(f"Original Date Range: {start_date} to {end_date}")

        # Slice the data arrays by the specified date range (already formatted)
        data_arrays = [da.sel(time=slice(*self.date_range)) for da in data_arrays]

        # Print and update the date range format for each DataArray after slicing
        for da in data_arrays:
            # Print the updated date range
            start_date = da['time'].min()
            end_date = da['time'].max()
            #print(f"Sliced Date Range: {start_date} to {end_date}")

        dataset = xr.merge(data_arrays, join='inner')
        dataset = dataset.transpose('time', 'lat', 'lon').compute()
        # Select date range
        #dataset = dataset.sel(time=slice(*self.date_range))
        return dataset
    
    def _get_lsm(self):
        """Return land-sea mask. """
        lsm = self.dataset[self.vars[0]].isel(time=0).isnull().astype(np.float32)
        lsm.name = 'lsm'
        del lsm['month']
        del lsm['time']
        lsm = lsm.data
        return lsm

    def _to_array(self, dataset):
        """Convert dataset to torch Tensor and normalize it."""
        np_arr = xr.concat([dataset[var] for var in self.vars], dim='variables').values # (variable, time, lat, lon)
        np.nan_to_num(np_arr, copy=False, nan=0.0)
        self.data = torch.from_numpy(np_arr).float().share_memory_()
        self.coords = {coord: dataset.coords[coord].values for coord in dataset.coords}

        if self.normalizer is not None:
            assert self.vars == self.normalizer.vars
            self.data = self.normalizer(self.data.unsqueeze(0)).squeeze(0)  # (variable, time, lat, lon)
            self.data.share_memory_()

    def __len__(self):
        return self.data.shape[1] - self.sequence_length + 1

    def __getitem__(self, idx):
        timeseries = self.data[self._channel_indices, idx : idx + self.sequence_length]
        month = self.coords['month'][idx : idx + self.sequence_length]
        month = month - 1  # 1-indexed to 0-indexed
        sample = {
            'x': timeseries,
            'month': month,
            'idx_time': torch.arange(idx, idx + self.sequence_length, dtype=torch.int)
        }
        if self.transform:
            sample = self.transform(sample)

        return sample


def load_reanalysis(root_dir, vars, sequence_length, split, 
                    normalizer=None, transform=None):
    """Load reanalysis dataset, either ORAS5 or CERA-20C.
    
    Args:
        root_dir (str): Folder containing netcdf files.
        vars (list): Variables.
        sequence_length (int): Sequence length.
        split (str): Split of dataset, either "train" or "val".
        lsm_path (str): Path to common land-sea mask between datasets.
        normalizer (_type_, optional): Normalizer of data. For split='val' a normalizer needs to be set. Defaults to None.
        transform (_type_, optional): Pytorch transfrom . Defaults to None.
    Returns:
        NetcdfReanalysisForecastingDataset: Dataset object.
    """
    if split == 'train':
        date_range = ('1983-01-16', "2022-09-16")
        #date_range = ('1983-01-16', "2000-01-16")
    elif split == 'all':
        date_range = (None, None)
    elif split == 'val':
        date_range = ('2000-01-01', None) 
        if normalizer is None:
            raise ValueError('normalizer must be provided for val split')
    else:
        raise ValueError(f'Invalid split: {split}')

    dataset = NetcdfReanalysisDataset(root_dir, vars, sequence_length,
                                      date_range, normalizer, transform)

    return dataset



# Function to create subsamples for cross-validation
def create_subsamples(dataloader, indices, cfg):
    subsample_loaders = []
    
    for i in range(len(indices) - 1):
        # Get start and end indices for the subset
        start_idx, end_idx = indices[i], indices[i + 1]
        
        # Create subset of the dataset
        subset_indices = list(range(start_idx, end_idx))
        subset = Subset(dataloader.dataset, subset_indices)
        
        # Create DataLoader for the subset
        subset_loader = DataLoader(
            subset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=cfg['num_workers']
        )
        
        subsample_loaders.append(subset_loader)
    
    return subsample_loaders

if __name__ == '__main__':
    import time

    root = '/mnt/qb/goswami/processed_data/enso/CESM2/historical'

    ts = time.time()
    # ds = NetcdfGeoDataset(f'{root}/cmip6_1001_001', ['ssha'], 12)
    # ds = cesm_lens_dataset([f'{root}/cmip6_1001_001', f'{root}/cmip6_1021_002'], ['ssha'], 12)
    ds = load_cesm_lens(root, ['ssha'], 12, 'val', 1, 2)

    print(f'Loaded dataset in {time.time() - ts:.2f} seconds')

    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    print(len(ds), len(dl))
    for batch in dl:
        print(batch.shape)
        break
