import xarray as xr

data_path = "/mnt/qb/datasets/STAGING/goswami/CMIP6_LENS/CESM2/piControl/tauy_1_2_grid/"

# Assuming you have two datasets, ds1 and ds2
ds1 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.070001-079912tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
ds2 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.080001-089912tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
ds3 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.090001-099912tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
ds4 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.100001-109912tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
ds5 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.110001-119912tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
ds6 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.120001-129912tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
ds7 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.130001-139912tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
ds8 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.140001-149912tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
ds9 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.150001-159912tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
ds10 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.160001-169912tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
ds11 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.170001-179912tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
ds12 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.180001-189912tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
ds13 = xr.open_dataset(data_path + "CMIP6-piControl.TAUY.190001-200012tauya_lat-31_33_lon90_330_gr1.0_2.0.nc")
# Merge the datasets
combined_ds = xr.concat([ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9, ds10, ds11, ds12, ds13], dim="time")

# Check the combined dataset
print(combined_ds)

# Optionally, save the combined dataset to a new NetCDF file
combined_ds.to_netcdf(data_path + "/concatenated/cesm2_picontrol_tauy_concatenated.nc")
print("Saved concatenated dataset")