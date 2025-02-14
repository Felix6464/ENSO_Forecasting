''' Preprocess CESM2-LENS data. 

@Author  :   Jakob Schlör 
@Time    :   2023/08/09 16:18:54
@Contact :   jakob.schloer@uni-tuebingen.de
'''
# %%
# Imports and parameters
# ======================================================================================
import os, argparse, nc_time_axis
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from s2aenso.utils import preproc

PATH = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--dirpath', default='',
                    type=str, help='Folder or filepath.')
parser.add_argument('-prefix', '--prefix', default='',
                    type=str, help='Folder or filepath.')
parser.add_argument('-name', '--name', default='',
                    type=str, help='New name of the dataset.')
parser.add_argument('-var', '--var', default='TEMP',
                    type=str, help='Variable name.')
parser.add_argument('-cpus', '--n_cpus', default=8,
                    type=int, help='Number of cpus.')
parser.add_argument('-outpath', '--outpath', default='',
                    type=str, help='Output filename.')
parser.add_argument('-level', '--level', default=0, type=int, help='Level of the ocean temperature variable.')
config = vars(parser.parse_args())

config['lon_range']=[130, 290]
config['lat_range']=[-31, 33]
config['grid_step_lat']=1.0
config['grid_step_lon']=1.0
config['climatology']='month'
config['name'] = config['name'] + f'_level{config["level"]}'

if config["var"] == "uflx":
    rename_vars = {'uflx': 'taux'}
elif config["var"] == "vflx":
    rename_vars = {"vflx": "tauy"}
# %%
# Load and merge data
# ======================================================================================
print("Load and merge data")
var = config['var']
nc_files = [os.path.join(config['dirpath'],file) for file in os.listdir(config['dirpath']) if file.endswith('.nc')]

da_arr = []
#da_arr.append(xr.open_mfdataset(nc_files)[var])
for file in nc_files:
    ds = xr.open_dataset(file)
    #print(f"File: {file}, nav_lon: {ds['nav_lon'].values}")
    #print(f"File: {file}, nav_lat: {ds['nav_lat'].values}")
    da_arr.append(ds[var])


orig_da = xr.concat(da_arr, dim='time', coords='all')
orig_da = orig_da.transpose('time', 'lat', 'lon')





# %%
# Flatten and remove NaNs 
# ======================================================================================
#print("Original data:", orig_da)

n_time, n_lat, n_lon = orig_da.shape         #2, 384, 320
orig_data_flat = orig_da.values.reshape((n_time, -1))

# Get latitude and longitude values
orig_lats = orig_da['lat'].values  # Shape: (360,)
orig_lons = orig_da['lon'].values  # Shape: (418,)

# Create 2D meshgrid for latitudes and longitudes to match data shape
orig_lons_2d, orig_lats_2d = np.meshgrid(orig_lons, orig_lats)

# Flatten the 2D grids to align with the flattened data
orig_lats = orig_lats_2d.ravel()  # Shape: (360 * 418,)
orig_lons = orig_lons_2d.ravel()  # Shape: (360 * 418,)

# Check if the flattened latitude and longitude arrays have matching lengths
if len(orig_lats) != orig_data_flat.shape[1] or len(orig_lons) != orig_data_flat.shape[1]:
    raise ValueError("Mismatch between the number of lat/lon points and the flattened data array")

# Remove NaNs
#idx_not_nan = np.argwhere(~np.isnan(orig_lats) & ~np.isnan(orig_lons))
#orig_data_filtered = orig_data_flat[:, idx_not_nan]
#orig_lats_filtered = orig_lats[idx_not_nan]
#orig_lons_filtered = orig_lons[idx_not_nan]


# %%
# Interpolate to regular grid
# ======================================================================================
def interp_to_regular_grid(data, timeidx, orig_lats, orig_lons,
                           lat_grid, lon_grid, method='linear'):
    """Interpolation of dataarray to a new set of points.

    Args:
        data (np.ndarray): Data on original grid but flattened
            of shape (n_time, n_flat) or (n_flat)
        timeidx (int): Index of time in case of parralelization.
        points_origin (np.ndarray): Array of origin locations.
            of shape (n_flat) 
        points_grid (np.ndarray): Array of locations to interpolate on.
            of shape (n_flat) 

    Returns:
        i (int): Index of time
        values_grid_flat (np.ndarray): Values on new points.
    """
    if timeidx is None:
        orig_data = data
    else:
        orig_data = data[timeidx, :]

    values_grid = griddata(
        (orig_lats.ravel(), orig_lons.ravel()),
        orig_data.ravel(),
        (lat_grid, lon_grid),
        method=method
    )
    return values_grid, timeidx

# Create array of new grid points 
new_lats = np.arange(*config['lat_range'], config['grid_step_lat'])
new_lons = np.arange(*config['lon_range'], config['grid_step_lon'])
lon_grid, lat_grid = np.meshgrid(new_lons, new_lats)

# Interpolate each time step in parallel
print("Interpolate to regular grid")
n_processes = n_time
results = Parallel(n_jobs=config['n_cpus'])(
    delayed(interp_to_regular_grid)(
        orig_data_flat, timeidx, orig_lats, orig_lons,
        lat_grid, lon_grid, method='linear'
    ) for timeidx in tqdm(range(n_processes))
)

# Store interpolated dataarray 
interpolated_data = np.array([r[0] for r in results])
timeids = np.array([r[1] for r in results])
timepoints = orig_da['time'].values[timeids]

interp_da = xr.DataArray(
    data=np.array(interpolated_data),
    dims=['time', 'lat', 'lon'],
    coords=dict(time=timepoints, lat=new_lats, lon=new_lons),
    name=rename_vars[orig_da.name] if orig_da.name in rename_vars.keys() else orig_da.name
)
# %%
# Detrend and compute anomalies 
# ======================================================================================
print("Compute anomalies:", flush=True)
da_detrend = preproc.detrend(interp_da, dim='time', deg=1)
da_anomaly, climatology = preproc.compute_anomalies(da_detrend, group='month')
da_anomaly.name = f"{interp_da.name}a"
climatology.name = f"{interp_da.name}c"

# %%
# Save to file 
# ======================================================================================
print("Save target to file!")
outpath = config['outpath']
if not os.path.exists(outpath):
    print(f"Create directoty {outpath}", flush=True)
    os.makedirs(outpath)
varname = da_anomaly.name
outfname =(os.path.join(config['outpath'], config['name'])  
      + f"{varname}_lat{'_'.join(map(str, config['lat_range']))}"
      + f"_lon{'_'.join(map(str,config['lon_range']))}_gr{config['grid_step_lat']}_{config['grid_step_lon']}_level{config['level']}.nc")

da_anomaly.to_dataset().to_netcdf(outfname)
varname = climatology.name
# Save climatology
#outfname = os.path.join(config['outpath'], config['name']) + f"climatology_{varname}c" + f"_lat{'_'.join(map(str, config['lat_range']))}" + f"_lon{'_'.join(map(str,config['lon_range']))}" + f"_gr{config['grid_step_lat']}_{config['grid_step_lon']}_level{config['level']}.nc"
#climatology.to_dataset().to_netcdf(outfname)
# %%
