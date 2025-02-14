import xarray as xr

data_path = "/mnt/qb/datasets/STAGING/goswami/CMIP6_LENS/CESM2/piControl/temp_ocean/"
years = ["year_070001-079912", "year_080001-089912", "year_090001-099912", "year_100001-109912", "year_110001-119912", "year_120001-129912", "year_130001-139912", "year_140001-149912", "year_150001-159912",
          "year_160001-169912", "year_170001-179912", "year_180001-189912", "year_190001-200012"]
levels = [0, 1, 3, 5, 8, 11, 14]

for level in levels:
    # Assuming you have two datasets, ds1 and ds2
    ds1 = xr.open_dataset(data_path + years[0] + f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[0].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    ds2 = xr.open_dataset(data_path + years[1] + f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[1].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    ds3 = xr.open_dataset(data_path + years[2] + f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[2].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    ds4 = xr.open_dataset(data_path + years[3] +  f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[3].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    ds5 = xr.open_dataset(data_path + years[4] +  f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[4].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    ds6 = xr.open_dataset(data_path + years[5] +  f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[5].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    ds7 = xr.open_dataset(data_path + years[6] +  f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[6].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    ds8 = xr.open_dataset(data_path + years[7] +  f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[7].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    ds9 = xr.open_dataset(data_path + years[8] +  f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[8].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    ds10 = xr.open_dataset(data_path + years[9] +  f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[9].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    ds11 = xr.open_dataset(data_path + years[10] +  f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[10].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    ds12 = xr.open_dataset(data_path + years[11] +  f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[11].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    ds13 = xr.open_dataset(data_path + years[12] +  f"/CMIP6-piControl.TEMP.lower_levels._level{level}{years[12].split("_")[1]}temp_ocn_{level}a_lat-31_33_lon130_290_gr1.0_1.0_level{level}.nc")
    # Merge the datasets
    combined_ds = xr.concat([ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9, ds10, ds11, ds12, ds13], dim="time")

    # Check the combined dataset
    print(combined_ds)

    # Optionally, save the combined dataset to a new NetCDF file
    combined_ds.to_netcdf(data_path + "processed/" + f"CMIP6-piControl.TEMP.1_1g_level_{level}_year_700_2000.nc")