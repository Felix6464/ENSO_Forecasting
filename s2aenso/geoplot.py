''' Plot geospatial data.

@Author  :   Jakob Schlör 
@Time    :   2022/08/18 10:56:26
@Contact :   jakob.schloer@uni-tuebingen.de
'''

import sys
import os
import numpy as np
import xarray as xr
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy as ctp
import torch
from torch.utils.data import Dataset, DataLoader

PATH = os.path.dirname(os.path.abspath(__file__))
plt.style.use(PATH + "/../paper.mplstyle")


def create_map_plot(ax=None, ctp_projection='PlateCarrree',
                    central_longitude=0,
                    gridlines_kw=dict(
                        draw_labels=True, dms=True, x_inline=False, y_inline=False
                    )):
    """Generate cartopy figure for plotting.

    Args:
        ax ([type], optional): [description]. Defaults to None.
        ctp_projection (str, optional): [description]. Defaults to 'PlateCarrree'.
        central_longitude (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: [description]

    Returns:
        ax (plt.axes): Matplotplib axes object.
    """
    if ax is None:
        # set projection
        if ctp_projection == 'Mollweide':
            proj = ctp.crs.Mollweide(central_longitude=central_longitude)
        elif ctp_projection == 'EqualEarth':
            proj = ctp.crs.EqualEarth(central_longitude=central_longitude)
        elif ctp_projection == 'Robinson':
            proj = ctp.crs.Robinson(central_longitude=central_longitude)
        elif ctp_projection == 'PlateCarree':
            proj = ctp.crs.PlateCarree(central_longitude=central_longitude)
        else:
            raise ValueError(
                f'This projection {ctp_projection} is not available yet!')

        fig, ax = plt.subplots()
        ax = plt.axes(projection=proj)

    # axes properties
    ax.coastlines()
    gl = ax.gridlines(**gridlines_kw)
    ax.add_feature(ctp.feature.RIVERS)
    ax.add_feature(ctp.feature.BORDERS, linestyle=':')
    return ax, gl


def plot_map(dmap, ax=None, vmin=None, vmax=None, eps=0.1,   
             cmap='RdBu_r', centercolor=None, bar='discrete', add_bar=True, 
             ctp_projection='PlateCarree', transform=None, central_longitude=0,
             kwargs_pl=None,
             kwargs_cb=dict(orientation='horizontal', shrink=0.8, extend='both'),
             kwargs_gl=dict(auto_inline=False, draw_labels=True, dms=True)
             ):
    """Simple map plotting using xArray.

    Args:
        dmap ([type]): [description]
        central_longitude (int, optional): [description]. Defaults to 0.
        vmin ([type], optional): [description]. Defaults to None.
        vmax ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        fig ([type], optional): [description]. Defaults to None.
        color (str, optional): [description]. Defaults to 'RdBu_r'.
        bar (bool, optional): [description]. Defaults to True.
        ctp_projection (str, optional): [description]. Defaults to 'PlateCarree'.
        label ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if kwargs_pl is None:
        kwargs_pl = dict()

    # create figure
    ax, gl = create_map_plot(ax=ax, ctp_projection=ctp_projection,
                         central_longitude=central_longitude,
                         gridlines_kw=kwargs_gl)


    # choose symmetric vmin and vmax
    if vmin is None and vmax is None:
         vmin = dmap.min(skipna=True)
         vmax = dmap.max(skipna=True)
         vmax = vmax if vmax > (-1*vmin) else (-1*vmin)
         vmin = -1*vmax

    # Select colormap
    if bar == 'continuous':
        cmap = plt.get_cmap(cmap)
        kwargs_pl['vmin'] = vmin 
        kwargs_pl['vmax'] = vmax
    elif bar == 'discrete':
        if 'norm' not in kwargs_pl:
            eps = (vmax-vmin)/10 if eps is None else eps
            bounds = np.arange(vmin, vmax+eps-1e-5, eps)
            # Create colormap
            n_colors = len(bounds)+1
            cmap = plt.get_cmap(cmap, n_colors)
            colors = np.array([mpl.colors.rgb2hex(cmap(i)) for i in range(n_colors)])
            # Set center of colormap to specific color
            if centercolor is not None:
                idx = [len(colors) // 2 - 1, len(colors) // 2]
                colors[idx] = centercolor 
            cmap = mpl.colors.ListedColormap(colors)
            kwargs_pl['norm'] = mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend='both')
        else:
            cmap = plt.get_cmap(cmap)
    else:
        raise ValueError(f"Specified bar={bar} is not defined!")

    if transform is None:
        transform = ctp.crs.PlateCarree(central_longitude=central_longitude)

    # plot map
    im = ax.pcolormesh(
        dmap.coords['lon'], dmap.coords['lat'], dmap.data,
        cmap=cmap, transform=transform,
        **kwargs_pl
    )

    # set colorbar
    if add_bar:
        if 'label' not in list(kwargs_cb.keys()):
            kwargs_cb['label'] = dmap.name
        cbar = plt.colorbar(im, ax=ax, **kwargs_cb)
    else:
        cbar = None

    return {'ax': ax, "im": im, 'gl': gl, 'cb': cbar}



