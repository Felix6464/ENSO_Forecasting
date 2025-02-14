import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

#1603 1s in the mask

# Load data
var = 'ssha'
paths = {
    'piControl': f"/home/goswami/gkd235/deeps2a-enso/data/processed_data/cesm2_lens/piControl/temp_ocean_1_2_grid/year_070001-079912/CMIP6-piControl.TEMP.070001-079912_level0temp_ocn_0a_lat-31_33_lon90_330_gr1.0_2.0_level0.nc",
    # Additional paths can be added here...
}

datasets = {}
for key, path in paths.items():
    datasets[key] = xr.open_dataset(path)

# Create common land-sea mask
mask = xr.ones_like(datasets['piControl'].isel(time=0))
for key, da in datasets.items():
    mask *= xr.where(np.isnan(da.isel(time=0)), 0, 1)
    
# Invert mask, i.e. 1 land, 0 ocean
lsm = xr.where(mask, 0, 1).drop_vars(['time', 'month'])

if isinstance(lsm, xr.Dataset):
    lsm = lsm.to_array()  # Convert to DataArray

# Sum all 1s and extract as a Python integer
count_ones = int(lsm.sum().values)

# Print the result
print("LSM:\n", lsm)
print("Number of 1s in the mask:", count_ones)

# Convert to DataArray if needed
if isinstance(lsm, xr.Dataset):
    lsm = lsm.to_array(name='lsm').squeeze()  # Convert to DataArray and assign name

# Set attributes
lsm.attrs = {'description': 'Common land-sea mask for CESM2-piControl, CESM2-LENS, ORAS5, and CERA-20C. LSM is zeros over oceans and ones over land.'}

# Save to file
lsm.to_netcdf("/home/goswami/gkd235/deeps2a-enso/data/processed_data/enso_data_pacific/land_sea_mask_common_1_2_grid.nc")
print("Common land-sea mask saved to file.")