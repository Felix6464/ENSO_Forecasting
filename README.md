# deeps2a-enso
Deep learning models for seasonal-to-annual forecasting of ENSO.

## Install

1. Create environment and install required packages
```
    mamba create -n deeps2aEnv
    mamba install pytorch torchvision torchaudio pytorch-cuda=<yourcudaversion> -c pytorch -c nvidia
    mamba env update -n deeps2aEnv -f condaenv.yml
```
2. Install local repository as pip package by running 'pip install -e .' in the root directory

## Suggested data structure  

./data
└── processed_data
    ├── cera-20c
    │   ├── ssha_lat-31_33_lon130_290_gr1.0.nc
    │   └── ssta_lat-31_33_lon130_290_gr1.0.nc
    ├── cesm2_lens
    |   ├── historical
    |   │   ├── cmip6_1001_001
    |   │   ├── cmip6_1021_002
    |   │   ├── ... 
    |   └── piControl
    |       ├── b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssha_lat-31_33_lon130_290_gr1.0.nc
    |       └── b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssta_lat-31_33_lon130_290_gr1.0.nc
    ├── land_sea_mask_common.nc
    └── oras5
        ├── ssha_lat-31_33_lon130_290_gr1.0.nc
        └── ssta_lat-31_33_lon130_290_gr1.0.nc

## Contribute and Guidelines

You are welcome to contribute. Please keep in mind the following guidelines:

- Datasets are stored in the `/data` folder (for large datasets store a link to the original location in the data folder)
- Outputs and plots are stored in `/output`
- Trained models are stored in `/models`
- Please use relative paths in all scripts only
- Comment your code!!! Use the [google docstring standard](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings).
- Please use the following linters:
	- pylint
