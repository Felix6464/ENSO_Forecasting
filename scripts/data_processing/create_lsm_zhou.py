import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Load data
paths = {
    'piControl': r"C:\\Users\\felix\\PycharmProjects\\deeps2a-enso\\data\\paper_data\\CMIP6_test.nc",
    # Additional paths can be added here...
}

datasets = {}
for key, path in paths.items():
    datasets[key] = xr.open_dataset(path)

# Create common land-sea mask
dataset = datasets['piControl'].isel(n_model=0, n_mon=0)
dataset = dataset[["tauxNor"]]
print("Dataset : ", dataset)
mask = xr.ones_like(dataset)
for key, da in datasets.items():
    mask *= xr.where(np.isnan(da[["tauxNor"]].isel(n_mon=0, n_model=0)), 0, 1)
    
print("Mask : ", mask)
# Invert mask, i.e. 1 land, 0 ocean
lsm = xr.where(mask, 0, 1)
# Count the number of 1s
if isinstance(lsm, xr.Dataset):
    lsm = lsm.to_array()  # Convert to DataArray

# Sum all 1s and extract as a Python integer
count_ones = int(lsm.sum().values)
#2601 1s in the mask

# Print the result
print("LSM:\n", lsm)
print("Number of 1s in the mask:", count_ones)


# Convert to DataArray if needed
if isinstance(lsm, xr.Dataset):
    lsm = lsm.to_array(name='lsm').squeeze()  # Convert to DataArray and assign name

# Set attributes
lsm.attrs = {'description': 'Common land-sea mask for CESM2-piControl, CESM2-LENS, ORAS5, and CERA-20C. LSM is zeros over oceans and ones over land.'}

print("LSM : ", lsm)
# Save to file
lsm.to_netcdf(r"C:\\Users\\felix\\PycharmProjects\\deeps2a-enso\\data\\paper_data\\lsm_zhou.nc")
print("Common land-sea mask saved to file.")