''' Process CERA-20C data for ENSO prediction. 

@Author  :   Jakob Schlör 
@Time    :   2024/02/02 10:21:15
@Contact :   jakob.schloer@uni-tuebingen.de
'''
# %%
import os
import numpy as np
import xarray as xr

from s2aenso.utils import preproc

# %%
# Load data
# ======================================================================================
config = {'path': '/mnt/qb/goswami/data/CERA20c/month/ssh/member_mean_r1x1/ssh_cera20c_1901-2009_r1x1.nc',
          'var': 'ssh',
          'outpath': '../../data/processed_data/cera-20c/'}
config['lon_range']=[130, 290]
config['lat_range']=[-31, 33]
config['grid_step']=1.0
config['climatology']='month'

orig_da = xr.open_dataset(config['path'])[config['var']]

# %%
# Renaming dimensions and sorting
# ======================================================================================
orig_da['lon'] = preproc.lon_to_360(orig_da['lon'])
orig_da = orig_da.sortby('lat').sortby('lon')

# %%
# Interpolate 
# ======================================================================================
# Create array of new grid points 
new_lats = np.arange(*config['lat_range'], config['grid_step'])
new_lons = np.arange(*config['lon_range'], config['grid_step'])

da = orig_da.interp(dict(lon=new_lons, lat=new_lats), method='nearest')

# %%
# Detrend and compute anomalies 
# ======================================================================================
print("Compute anomalies using fair sliding:", flush=True)
da_anomaly = preproc.fair_sliding_anomalies(da, window_size_year=30, 
                                            group='month', detrend=True)    

# %%
# Save to file 
# ======================================================================================
print("Save target to file!")
outpath = config['outpath']
if not os.path.exists(config['outpath']):
    print(f"Create directoty {config['outpath']}", flush=True)
    os.makedirs(config['outpath'])

outfname =(config['outpath'] 
      + f"/{da_anomaly.name}_lat{'_'.join(map(str, config['lat_range']))}"
      + f"_lon{'_'.join(map(str,config['lon_range']))}_gr{config['grid_step']}.nc")

da_anomaly.to_netcdf(outfname)
# %%
