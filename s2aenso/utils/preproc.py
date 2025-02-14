'''Collection of functions to preprocess xarray climate data.

@Author  :   Jakob Schlör 
@Time    :   2024/02/02 14:25:50
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import cftime
import numpy as np
import xarray as xr


def lon_to_180(longitudes):
    """Convert longitudes from [0, 360] to [-180, 180]."""
    return np.where(longitudes > 180, longitudes - 360, longitudes)

def lon_to_360(longitudes):
    """Convert longitudes from [-180, 180] to [0, 360]."""
    return np.where(longitudes < 0, longitudes + 360, longitudes)

def check_dimensions(ds, sort=True):
    """
    Checks whether the dimensions are the correct ones for xarray!
    """
    dims = list(ds.dims)

    rename_dic = {
        'longitude': 'lon',
        'latitude': 'lat',
        'nav_lon': 'lon',
        'nav_lat': 'lat'
    }
    for c_old, c_new in rename_dic.items():
        if c_old in dims:
            print(f'Rename:{c_old} : {c_new} ', flush=True)
            ds = ds.rename({c_old: c_new})
            dims = list(ds.dims)

    # Check for dimensions
    clim_dims = ['lat', 'lon']
    for dim in clim_dims:
        if dim not in dims:
            raise ValueError(
                f"The dimension {dim} not consistent with required dims {clim_dims}!")

    # If lon from 0 to 360 shift to -180 to 180
    if max(ds.lon) > 180:
        print("Shift longitude!", flush=True)
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))

    if sort:
        print('Sort longitudes and latitudes in ascending order, respectively', flush=True)
        ds = ds.sortby('lon')
        ds = ds.sortby('lat')

    return ds


def time_average(ds, group='1D'):
    """Downsampling of time dimension by averaging.

    Args:
    -----
    ds: xr.dataFrame 
        dataset
    group: str
        time group e.g. '1D' for daily average from hourly data
    """
    ds_average = ds.resample(time=group, label='left').mean(skipna=True)

    # Shift time to first of month
    if group == '1M':
        new_time = ds_average.time.data + np.timedelta64(1, 'D')
        new_coords = {}
        for dim in ds_average.dims:
            new_coords[dim] = ds_average[dim].data
        new_coords['time'] = new_time
        ds_average = ds_average.assign_coords(coords=new_coords)

    return ds_average


def set_grid(ds, step_lat=1, step_lon=1,
             lat_range=None, lon_range=None):
    """Interpolate grid.

    Args:
        ds (xr.Dataset): Dataset or dataarray to interpolate.
            Dataset is only supported for grid_type='mercato'.
        step_lat (float, optional): Latitude grid step. Defaults to 1.
        step_lon (float, optional): Longitude grid step. Defaults to 1.

    Returns:
        da (xr.Dataset): Interpolated dataset or dataarray.
        grid (dict): Grid used for interpolation, dict(lat=[...], lon=[...]).
    """
    lat_min = ds['lat'].min().data if lat_range is None else lat_range[0] 
    lat_max = ds['lat'].max().data if lat_range is None else lat_range[1] 
    lon_min = ds['lon'].min().data if lon_range is None else lon_range[0] 
    lon_max = ds['lon'].max().data if lon_range is None else lon_range[1] 
    init_lat = np.arange(
        lat_min, (lat_max + step_lat), step_lat
    )
    init_lon = np.arange(
        lon_min, lon_max, step_lon
    )
    grid = {'lat': init_lat, 'lon': init_lon}
    # Interpolate
    da = ds.interp(grid, method='linear')

    return da, grid


def cut_map(ds, lon_range=None, lat_range=None, shortest=True):
    """Cut area. 
    
    Args:
        da (xr.Dataset): 
        lon_range (list): Range of longitudes [min, max]. Defaults to None.
        lat_range (list): Range of latitudes [min, max]. Defaults to None.
        shortest (boolean): Use shortest range in longitude, 
            eg. -170, 170 range contains all points from 170 to 180, -180 to -170,
            and not all between -170 and 170. Default is True.
    Return:
        ds (xr.Dataset): Dataset cut to range
    """
    if lon_range is not None:
        if (max(lon_range) - min(lon_range) <= 180) or shortest is False:
            ds = ds.sel(
                lon=slice(np.min(lon_range), np.max(lon_range)),
                lat=slice(np.min(lat_range), np.max(lat_range))
            )
        else:
            # To account for areas that lay at the border of -180 to 180
            ds = ds.sel(
                lon=ds.lon[(ds.lon < min(lon_range)) |
                           (ds.lon > max(lon_range))],
                lat=slice(np.min(lat_range), np.max(lat_range))
            )
    if lat_range is not None:
        ds = ds.sel(
            lat=slice(np.min(lat_range), np.max(lat_range))
        )

    return ds


def get_antimeridian_coord(lons):
    """Change of coordinates from normal to antimeridian."""
    lons = np.array(lons)
    lons_new = np.where(lons < 0, (lons + 180), (lons - 180))
    return lons_new


def set_antimeridian2zero(ds, roll=True):
    """Set the antimeridian to zero.

    Easier to work with the pacific then.
    """
    if ds['lon'].data[0] <= -100 and roll is True:
        # Roll data such that the dateline is not at the corner of the dataset
        print("Roll longitudes.", flush=True)
        ds = ds.roll(lon=(len(ds['lon']) // 2), roll_coords=True)

    # Change lon coordinates
    lons_new = get_antimeridian_coord(ds.lon)
    ds = ds.assign_coords(
        lon=lons_new
    )
    print('Set the dateline to the new longitude zero.', flush=True)
    return ds

def compute_anomalies(dataarray, group='dayofyear',
                      base_period=None):
    """Calculate anomalies.

    Parameters:
    -----
    dataarray: xr.DataArray
        Dataarray to compute anomalies from.
    group: str
        time group the anomalies are calculated over, i.e. 'month', 'day', 'dayofyear'
    base_period (list, None): period to calculate climatology over. Default None.

    Return:
    -------
    anomalies: xr.dataarray
    """
    if base_period is None:
        base_period = np.array(
            [dataarray.time.data.min(), dataarray.time.data.max()])

    climatology = dataarray.sel(time=slice(base_period[0], base_period[1])
                                ).groupby(f'time.{group}').mean(dim='time', skipna=True)
    anomalies = (dataarray.groupby(f"time.{group}")
                 - climatology)

    return anomalies, climatology


def detrend(da: xr.DataArray, dim: str='time', deg: int=1):
    """Detrend data by subtracting a linear fit.

    Args:
        da ([xr.DataArray]): Data to detrend.

    Returns:
        da_detrend (xr.DataArray): Detrended data.
        dim (str, optional): Dimension to detrend. Defaults to 'time'.
        deg (int, optional): Degree of polynomial fit. Defaults to 1.
    """
    coeff = da.polyfit(dim=dim, deg=deg)
    trend = xr.polyval(da[dim], coeff.polyfit_coefficients)

    da_detrend =  da - trend + trend.isel({dim: 0}) 

    return da_detrend

    
def fair_sliding_anomalies(da: xr.DataArray, window_size_year: int=30,
                           group: str ='month', detrend=False, climatology_inp=None) -> xr.DataArray:
    """Compute anomalies using the fair-sliding approach.

    Args:
        da (xr.DataArray): Dataarray. 
        window_size_year (int, optional): Window for fairsliding in years.
            Defaults to 30.
        group (str, optional): Group to compute climatology over.
            Defaults to 'month'.
        detrend (bool, optional): If True, a linear trend over the window is
            computed and removed before computing the anomalies.
            Defaults to False.

    Returns:
        da_anomalies (xr.DataArray): Anomalies of dimension (time, ...)
    """
    years = np.unique(da.time.dt.year)
    da_anomalies = []
    for y in years[window_size_year:]:
        window = da.sel(
            time=slice(f"{y-window_size_year}-01-01", f"{y-1}-12-31")
        )
        da_year = da.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))

        # Remove linear trend over window
        if detrend:
            p = window.polyfit(dim='time', deg=1)
            # Remove trend from window
            linear_fit = xr.polyval(window['time'], p.polyfit_coefficients)
            window =  window - linear_fit
            # Remove trend from year
            linear_fit_year = xr.polyval(da_year['time'], p.polyfit_coefficients)
            da_year = da_year - linear_fit_year

        # Remove climatology
        climatology = window.groupby(f'time.{group}').mean(dim='time')
        anomaly_year = (
            da_year.groupby(f'time.{group}') 
            - climatology
        )
        da_anomalies.append(anomaly_year)

    da_anomalies =  xr.concat(da_anomalies, dim='time')
    da_anomalies.name = f"{da.name}a"
    return da_anomalies, climatology


def select_months(ds, months=[12, 1, 2]):
    """Select only some months in the data.

    Args:
        ds ([xr.DataSet, xr.DataArray]): Dataset of dataarray
        months (list, optional): Index of months to select. 
                                Defaults to [12,1,2]=DJF.
    """
    ds_months = ds.sel(time=np.in1d(ds['time.month'], months))

    return ds_months


def select_time_snippets(ds, time_snippets):
    """Cut time snippets from dataset and concatenate them.
ra
    Parameters:
    -----------
    time_snippets: np.datetime64  (n,2)
        Array of n time snippets with dimension (n,2).

    Returns:
    --------
    xr.Dataset with concatenate times
    """
    ds_lst = []
    for time_range in time_snippets:
        ds_lst.append(ds.sel(time=slice(time_range[0], time_range[1])))

    ds_snip = xr.concat(ds_lst, dim='time')

    return ds_snip


def average_time_periods(ds, time_snippets):
    """Select time snippets from dataset and average them.

    Parameters:
    -----------
    time_snippets: np.datetime64  (n,2)
        Array of n time snippets with dimension (n,2).

    Returns:
    --------
    xr.Dataset with averaged times
    """
    ds_lst = []
    for time_range in time_snippets:
        temp_mean = ds.sel(time=slice(time_range[0], time_range[1])).mean('time')
        temp_mean['time'] = time_range[0] +  0.5 * (time_range[1] - time_range[0])
        ds_lst.append(temp_mean)

    ds_snip = xr.concat(ds_lst, dim='time')

    return ds_snip


def get_mean_time_series(da, lon_range, lat_range, time_roll=0):
    """Get mean time series of selected area.

    Parameters:
    -----------
    da: xr.DataArray
        Data
    lon_range: list
        [min, max] of longitudinal range
    lat_range: list
        [min, max] of latiduninal range
    """
    #da_area = cut_map(da, lon_range, lat_range, shortest=False)
    da_area = da.sel(lon=slice(lon_range[0], lon_range[1]), 
                     lat=slice(lat_range[0], lat_range[1]))
    ts_mean = da_area.mean(dim=('lon', 'lat'), skipna=True)
    ts_std = da_area.std(dim=('lon', 'lat'), skipna=True)
    if time_roll > 0:
        ts_mean = ts_mean.rolling(time=time_roll, center=True).mean()
        ts_std = ts_std.rolling(time=time_roll, center=True).mean()

    return ts_mean, ts_std

def is_datetime360(time):
    return isinstance(time, cftime._cftime.Datetime360Day)

def is_datetime(time):
    return isinstance(time, cftime._cftime.DatetimeNoLeap)

def time2timestamp(time):
    """Convert time to timestamp."""
    try:
        t = time[0]
    except:
        t = time

    if is_datetime(t):
        timestamp = cftime.date2num(t, 'days since 0001-01-01')
    else:
        timestamp = (t - np.datetime64('0001-01-01', 'ns')) / np.timedelta64(1, 'D') 
    return timestamp

def timestamp2time(timestamp, type=np.ndarray):
    """Convert timestamp to np.datetime64 object."""
    if type == cftime._cftime.DatetimeNoLeap:
        time = cftime.num2date(timestamp, 'days since 0001-01-01')
    else:
        time = (np.datetime64('0001-01-01', 'ns') 
                + timestamp * np.timedelta64(1, 'D') )
    return time

def time2idx(timearray, times):
    """Get index of t in times.
    
    Parameters:
        timearray (np.ndarray): Time array to search in.
        times (list): List of times.
    """
    idx = []
    for t in times:
        idx.append(np.argmin(np.abs(timearray - t)))
    return idx


def convert_datetime64(date: np.datetime64, dtype=cftime._cftime.DatetimeNoLeap):
    """Convert datetime64 dates into other time formats.

    Args:
        date (np.datetime64): Input date
        dtype (_type_, optional): Date type format.
            Defaults to cftime._cftime.DatetimeNoLeap.

    Returns:
        _type_: Date in other format.
    """
    date = np.array(date, dtype='datetime64[D]')
    years = date.astype('datetime64[Y]').astype(int) + 1970
    months = date.astype('datetime64[M]').astype(int) % 12 + 1
    days = (date - date.astype('datetime64[M]') + 1).astype(int)

    if date.size == 1:
        dates = cftime._cftime.DatetimeNoLeap(years, months, days)
    else:
        dates = []
        for i in range(date.size):
            if dtype == cftime._cftime.DatetimeNoLeap:
                dates.append(cftime._cftime.DatetimeNoLeap(years[i], months[i], days[i]))
    return np.array(dates)


def map2flatten(x_map: xr.Dataset) -> list:
    """Flatten dataset/dataarray and remove NaNs.

    Args:
        x_map (xr.Dataset/ xr.DataArray): Dataset or DataArray to flatten. 

    Returns:
        x_flat (xr.DataArray): Flattened dataarray without NaNs 
        ids_notNaN (xr.DataArray): Boolean array where values are on the grid. 
    """
    if type(x_map) == xr.core.dataset.Dataset:
        x_stack_vars = [x_map[var] for var in list(x_map.data_vars)]
        x_stack_vars = xr.concat(x_stack_vars, dim='var')
        x_stack_vars = x_stack_vars.assign_coords({'var': list(x_map.data_vars)})
        x_flatten = x_stack_vars.stack(z=('var', 'lat', 'lon')) 
    else:
        x_flatten = x_map.stack(z=('lat', 'lon'))

    # Flatten and remove NaNs
    if 'time' in x_flatten.dims:
        idx_notNaN = ~np.isnan(x_flatten.isel(time=0))
    else:
        idx_notNaN = ~np.isnan(x_flatten)
    x_proc = x_flatten.isel(z=idx_notNaN.data)

    return x_proc, idx_notNaN


def flattened2map(x_flat: np.ndarray, 
                  mask_flat: xr.DataArray, 
                  times: np.ndarray = None) -> xr.Dataset:
    """Transform flattened array without NaNs to gridded data with NaNs. 

    Args:
        x_flat (np.ndarray): Flattened array of size (n_times, n_points) or (n_points).
        mask_flat (xr.DataArray): Boolean dataarray of size (n_points).
        times (np.ndarray): Time coordinate of xarray if x_flat has time dimension.

    Returns:
        xr.Dataset: Gridded data.
    """
    mask_map = mask_flat.unstack().drop(('time', 'month'))
    idx_mask_flat = np.where(mask_flat)[0]
    n_flat = len(mask_flat)

    # For single time point
    if len(x_flat.shape) == 1:
        n_points = x_flat.shape
        x_map = np.ones((n_flat)) * np.nan
        x_map[idx_mask_flat] = x_flat

        # Create xarray
        x_map = xr.DataArray(
            data=x_map.reshape(*mask_map.shape),
            coords=mask_map.coords
        )
    else:
        n_times, n_points = x_flat.shape
        x_map = np.ones((n_times, n_flat)) * np.nan
        x_map[:, idx_mask_flat] = x_flat

        # Create xarray
        if times is None:
            times = np.arange(x_flat.shape[0]) 
        x_map = xr.DataArray(
            data=x_map.reshape((n_times, *mask_map.shape)),
            coords={'time': times, **mask_map.coords}
        )

    if 'var' in list(x_map.dims): 
        da_list = [xr.DataArray(x_map.isel(var=i), name=var) 
                   for i, var in enumerate(x_map['var'].data)]
        x_map = xr.merge(da_list, compat='override')
        x_map = x_map.drop(('var'))
    
    return x_map
