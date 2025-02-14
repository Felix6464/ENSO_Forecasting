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
import json

from s2aenso.utils import preproc

PATH = os.path.dirname(os.path.abspath(__file__))
#plt.style.use(PATH + "/../../paper.mplstyle")

# Function to load or create the dictionary
def load_or_create_dict(file_path):
    if os.path.exists(file_path):
        # Load existing dictionary from the file
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        # Return a new empty dictionary if the file doesn't exist
        return {}

# Function to save the dictionary back to the file
def save_dict_to_file(file_path, data_dict):
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=4)

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

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--dirpath', default='./',
                    type=str, help='Folder or filepath.')
parser.add_argument('-prefix', '--prefix', default='',
                    type=str, help='Folder or filepath.')
parser.add_argument('-var', '--var', default='SST',
                    type=str, help='Variable name.')
parser.add_argument('-cpus', '--n_cpus', default=8,
                    type=int, help='Number of cpus.')
parser.add_argument('-name', '--name', default='',
                    type=str, help='New name of the dataset.')
parser.add_argument('-outpath', '--outpath', default='./',
                    type=str, help='Folder to store file.')
parser.add_argument('-level', '--level', default=0, type=int, help='Level of the ocean temperature variable.')
parser.add_argument('-chunks', '--chunks', default=18, type=int)

config = vars(parser.parse_args())

config['lon_range']=[90, 330]
config['lat_range']=[-31, 33]
config['grid_step_lat']=1.0
config['grid_step_lon']=2.0
config['climatology']='month'
config['normalization']=None

if config["var"] == "TEMP":
    rename_vars = {'TEMP': f'temp_ocn_{config["level"]}'}
elif config["var"] == "TAUX":
    rename_vars = {'TAUX': 'taux'}
elif config["var"] == "TAUY":
    rename_vars = {'TAUY': 'tauy'}

ensemble_list = ["ensemble_1001_001", "ensemble_1011_001", "ensemble_1021_001", "ensemble_1031_002", "ensemble_1041_001", "ensemble_1051_003", "ensemble_1061_001", "ensemble_1071_004", "ensemble_1081_001", "ensemble_1091_005", "ensemble_1101_001",
                 "ensemble_1111_006", "ensemble_1121_001", "ensemble_1131_007", "ensemble_1141_001", "ensemble_1151_008", "ensemble_1161_001", "ensemble_1171_009", "ensemble_1181_001", "ensemble_1191_010",
                 "ensemble_1231_001", "ensemble_1251_001", "ensemble_1281_001", "ensemble_1301_001", "ensemble_1301_002", "ensemble_1301_010", "ensemble_1301_011"]

ensemble_list = ["ensemble_1281_002","ensemble_1281_003","ensemble_1281_004","ensemble_1281_005","ensemble_1281_006","ensemble_1281_007","ensemble_1281_008","ensemble_1281_009",
                 "ensemble_1281_010","ensemble_1281_011","ensemble_1281_012","ensemble_1281_013","ensemble_1281_014","ensemble_1281_015", "ensemble_1281_016","ensemble_1281_017",
                 "ensemble_1281_018","ensemble_1281_019","ensemble_1281_020","ensemble_1301_003","ensemble_1301_004","ensemble_1301_005","ensemble_1301_006", "ensemble_1301_007",
                 "ensemble_1301_008","ensemble_1301_009","ensemble_1301_012",
                 "ensemble_1301_013","ensemble_1301_014","ensemble_1301_015","ensemble_1301_016","ensemble_1301_017","ensemble_1301_018","ensemble_1301_019", "ensemble_1301_020"]

ensemble_list = ["ensemble_1251_002","ensemble_1251_003","ensemble_1251_004","ensemble_1251_005","ensemble_1251_006","ensemble_1251_007","ensemble_1251_008","ensemble_1251_009",
                 "ensemble_1251_010","ensemble_1251_011","ensemble_1251_012","ensemble_1251_013","ensemble_1251_014","ensemble_1251_015", "ensemble_1251_016","ensemble_1251_017",
                 "ensemble_1251_018","ensemble_1251_019","ensemble_1251_020",
                 "ensemble_1231_002","ensemble_1231_003","ensemble_1231_004","ensemble_1231_005","ensemble_1231_006","ensemble_1231_007","ensemble_1231_008","ensemble_1231_009",
                 "ensemble_1231_010","ensemble_1231_011","ensemble_1231_012","ensemble_1231_013","ensemble_1231_014","ensemble_1231_015", "ensemble_1231_016","ensemble_1231_017",
                 "ensemble_1231_018","ensemble_1231_019","ensemble_1231_020"]


ensemble_list = ["ensemble_1071_004", "ensemble_1091_005", "ensemble_1111_006", "ensemble_1131_007", "ensemble_1151_008", "ensemble_1171_009", "ensemble_1191_010",
                 "ensemble_1281_001", "ensemble_1301_001", "ensemble_1301_002", "ensemble_1301_010", "ensemble_1301_011"]
                
# %%
# Load and merge data
# ======================================================================================
print("Load and merge data")
var = config['var']

for ensemble_ in ensemble_list:

    dirpath = os.path.join(config['dirpath'], ensemble_)

    
    nc_files = [os.path.join(dirpath,file) for file in os.listdir(dirpath) if file.endswith('.nc')]

    chunk_size = config["chunks"]

    for i in range(0, len(nc_files), chunk_size):
        chunk_files = nc_files[i:i + chunk_size]

        # Load data from the current chunk
        da_arr = []
        orig_ds = xr.open_mfdataset(chunk_files, 
                                chunks={'time': 100},
                                combine='by_coords')
        #da_arr.append(orig_ds)
        #orig_da = xr.concat(da_arr, dim='time', coords='all')
        orig_da = orig_ds[var]

        if config["var"] == "TEMP" and config["level"] is not None:
            orig_da = orig_da.isel(z_t=config["level"])
            #orig_da = orig_da.isel(z_t_150m=config["level"])

        # Get the time range from the loaded chunk
        start_year = str(orig_da['time'].dt.year.min().item())
        end_year = str(orig_da['time'].dt.year.max().item())

        # Flatten and remove NaNs 
        n_time, n_lat, n_lon = orig_da.shape        #720 384 320
        orig_data_flat = orig_da.values.reshape((n_time, -1))
        orig_lats = orig_da['TLAT'].values.ravel()
        orig_lons = orig_da['TLONG'].values.ravel()

        # Remove NaNs
        mask_not_nan = ~np.isnan(orig_lats) & ~np.isnan(orig_lons)

        # Apply the mask to filter the data and coordinates
        orig_data_filtered = orig_data_flat[:, mask_not_nan]  # Apply mask to filter non-NaN data
        orig_lats_filtered = orig_lats[mask_not_nan]          # Filter corresponding latitudes
        orig_lons_filtered = orig_lons[mask_not_nan]          # Filter corresponding longitudes


        # Interpolate to regular grid
        new_lats = np.arange(*config['lat_range'], config['grid_step_lat'])
        new_lons = np.arange(*config['lon_range'], config['grid_step_lon'])
        lon_grid, lat_grid = np.meshgrid(new_lons, new_lats)

        # Parallel interpolation
        print("Interpolate to regular grid")
        n_processes = n_time
        results = Parallel(n_jobs=config['n_cpus'])(
            delayed(interp_to_regular_grid)(
                orig_data_filtered, timeidx, orig_lats_filtered, orig_lons_filtered,
                lat_grid, lon_grid, method='linear'
            ) for timeidx in tqdm(range(n_processes))
        )

        interpolated_data = np.array([r[0] for r in results])
        timeids = np.array([r[1] for r in results])
        timepoints = orig_da['time'].values[timeids]

        interp_da = xr.DataArray(
            data=np.array(interpolated_data),
            dims=['time', 'lat', 'lon'],
            coords=dict(time=timepoints, lat=new_lats, lon=new_lons),
            name=rename_vars[orig_da.name] if orig_da.name in rename_vars.keys() else orig_da.name
        )

        # Compute anomalies using fair sliding
        #print("Compute anomalies using fair sliding:", flush=True)
        #da_anomaly, climatology = preproc.fair_sliding_anomalies(interp_da, window_size_year=30, 
                                                    #group='month', detrend=True)

        print("Compute anomalies:", flush=True)
        da_detrend = preproc.detrend(interp_da, dim='time', deg=1)
        da_anomaly, climatology = preproc.compute_anomalies(da_detrend, group='month')
        da_anomaly.name = f"{interp_da.name}a"

        # Normalize data if needed
        if config['normalization'] is not None:
            print("Normalize data:", flush=True)
            normalizer = preproc.Normalizer(method=config['normalization'])
            da_anomaly = normalizer.fit_transform(da_anomaly)
            da_anomaly.attrs = normalizer.to_dict() 

        # Assuming you have calculated mean and std for da_anomaly
        mean_anomaly = float(da_anomaly.mean().values)
        std_anomaly = float(da_anomaly.std().values)

        # Define the output path for the dictionary file
        dict_file_path = os.path.join(config['outpath'], 'ensemble_stats.txt')

        # Load existing dictionary or create a new one
        ensemble_dict = load_or_create_dict(dict_file_path)
        
        # Add the mean and std for the current ensemble
        dict_key = f"{ensemble_}_level{config["level"]}"
        ensemble_dict[dict_key] = {
            'mean': mean_anomaly,
            'std': std_anomaly
        }

        # Save the updated dictionary back to the file
        save_dict_to_file(dict_file_path, ensemble_dict)

        # Continue with the existing saving process for da_anomaly
        print("Save target to file!")
        outpath = config['outpath']
        outpath = os.path.join(outpath, ensemble_)
        if not os.path.exists(outpath):
            print(f"Create directory {outpath}", flush=True)
            os.makedirs(outpath)

        prefix = ensemble_ + "_"
        varname = da_anomaly.name
        normalize = f"_norm-{config['normalization']}" if config['normalization'] is not None else ""
        outfname = (os.path.join(outpath, prefix)  
                    + f"{varname}_lat{'_'.join(map(str, config['lat_range']))}"
                    + f"_lon{'_'.join(map(str, config['lon_range']))}_gr{config['grid_step_lat']}_{config['grid_step_lon']}"
                    + f"_{start_year}-{end_year}{normalize}.nc")

        da_anomaly.to_netcdf(outfname)

    # %%
