# %%
import os, json, einops
import numpy as np
import xarray as xr
import torch
import pickle
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import zarr

from s2aenso.model import swinlstm, losses, geoformer
from s2aenso.utils import data, normalization, metric

# Set device and path
device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device: {device}')
PATH = os.path.dirname(os.path.abspath(__file__))


# ============================================================================================================================================================================
# Run configuration
# ============================================================================================================================================================================


datasets = ["godas", "cesm2_picontrol", "oras5", "cesm2_lens"]

#datasets = ["cesm2_picontrol"]
#### CHECK BEFORE EXECUTING ####

model_num = 835194
run_specification = "decay"
lens = True
model_path = 'best_oras5.pth' #'path_epoch10.pth'  #"best.pth" "cpkt.pth" "best_oras5.pth"
print("Modelpath: ", model_path)




# ============================================================================================================================================================================
# Load model and data
# ============================================================================================================================================================================

print("Modelnumber : ", model_num)
print("Run specification", run_specification)
modelpath = PATH + f"/../../models/basemodel_piControl/vit_piControl_{model_num}/" if not lens else PATH + f"/../../models/basemodel_lens/vit_lens_{model_num}/"
with open(os.path.join(modelpath, "config.json")) as file:
    cfg = json.load(file)

if cfg["probabilistic"]:
    from s2aenso.model import geoformer_probabilistic as geoformer


# Selecting the data directories
if cfg["temp_ocean"]:
    if cfg["1_2_grid"]:
        cfg["data_dir"] = PATH + '/../../data/processed_data/cesm2_lens/piControl/temp_ocean_1_2_grid/processed/'
        cfg["lens_dir"] = PATH + '/../../data/processed_data/cesm2_lens/historical_levels/temp_ocean/1_2_grid/final_ensembles/'
        cfg["oras5_dir"] = PATH + '/../../data/processed_data/enso_data_pacific/oras5/temp_ocean/1_2_grid/'
        cfg["godas_dir"] = PATH + '/../../data/processed_data/enso_data_pacific/godas/temp_ocean/1_2_grid/'
    else:
        cfg["data_dir"] = PATH + '/../../data/processed_data/cesm2_lens/piControl/temp_ocean/processed/'
        cfg["lens_dir"] = PATH + '/../../data/processed_data/cesm2_lens/historical_levels/temp_ocean/1_1_grid/final_ensembles/'
        cfg["oras5_dir"] = PATH + '/../../data/processed_data/enso_data_pacific/oras5/temp_ocean/1_1_grid/'
        cfg["godas_dir"] = PATH + '/../../data/processed_data/enso_data_pacific/godas/temp_ocean/1_1_grid/'
else:
    cfg["data_dir"] = PATH + '/../../data/processed_data/cesm2_lens/piControl/sst_ssh/'
    cfg["lens_dir"] = PATH + '/../../data/processed_data/cesm2_lens/historical_levels/sst_ssh/final_ensembles/'
    cfg["oras5_dir"] = PATH + '/../../data/processed_data/enso_data_pacific/oras5/sst_ssh/'

# Selecting the land sea mask
if cfg["1_2_grid"]:
    cfg["lsm_path"] = PATH + '/../../data/processed_data/enso_data_pacific/land_sea_mask_common_1_2_grid.nc'
else:
    cfg["lsm_path"] = PATH + '/../../data/processed_data/enso_data_pacific/land_sea_mask_common.nc'


# Normalizer
normalizer_lens = normalization.Normalizer_LENS(cfg['in_vars'], levels=cfg["temp_ocean"], low_levels=cfg["lower_levels"], grid=cfg["1_2_grid"])
normalizer_oras5 = None
normalizer_picontrol = None
normalizer_godas = None


# Initialize data loaders
val_data_picontrol = data.load_picontrol(root_dir = cfg['data_dir'],
                               vars = cfg['in_vars'], 
                               sequence_length= cfg['sequence_length_val'],
                               normalizer=normalizer_picontrol,
                               split= "val",
                               test=False)

val_dl_picontrol = torch.utils.data.DataLoader(val_data_picontrol, 
                                       batch_size=cfg['batch_size'], 
                                       shuffle=False, 
                                       drop_last=True,
                                       pin_memory=True,
                                       num_workers=cfg['num_workers'])

print("Length of PiControl val_dl: ", len(val_dl_picontrol))

val_data_oras5 = data.load_reanalysis(root_dir = cfg['oras5_dir'],
                                vars = cfg['in_vars'], 
                                sequence_length = cfg['sequence_length_val'],
                                normalizer=normalizer_oras5,
                                split= 'train')

val_dl_oras5 = torch.utils.data.DataLoader(val_data_oras5, 
                                    batch_size=cfg['batch_size'], 
                                    shuffle=False, 
                                    drop_last=True,
                                    pin_memory=True,
                                    num_workers=cfg['num_workers'])
print("Length of ORAS5 val_dl: ", len(val_dl_oras5))

val_data_godas = data.load_reanalysis(root_dir = cfg['godas_dir'],
                                vars = cfg['in_vars'], 
                                sequence_length = cfg['sequence_length_val'],
                                normalizer=normalizer_godas,
                                split= 'train')

val_dl_godas = torch.utils.data.DataLoader(val_data_godas, 
                                    batch_size=cfg['batch_size'], 
                                    shuffle=False, 
                                    drop_last=True,
                                    pin_memory=True,
                                    num_workers=cfg['num_workers'])
print("Length of GODAS val_dl: ", len(val_dl_godas))


val_data_lens = data.load_cesm_lens(root_dir=cfg['lens_dir'],
                               vars=cfg['in_vars'],
                               sequence_length=cfg['sequence_length_val'],
                               split='val',
                               rank=cfg['rank'],
                               size=cfg['world_size'],
                               num_directories_split=23,
                               max_split=25,
                               normalizer=normalizer_lens)

val_dl_lens = torch.utils.data.DataLoader(val_data_lens,
                                     batch_size=cfg['batch_size'],
                                     shuffle=False,
                                     drop_last=True,
                                     pin_memory=True,
                                     num_workers=cfg['num_workers'])

print("Length of LENS val_dl: ", len(val_dl_lens))

# Add common land sea mask
common_lsm = xr.open_dataset(cfg['lsm_path'])['lsm'].data
lsm = torch.as_tensor(common_lsm, device = device, dtype = torch.long)

# Model
cfg["device"] = device
cfg["cutout"] = 0
model = geoformer.Geoformer(config=cfg).to(device)

# Load checkpoint
path_ = model_path
model.load_state_dict(torch.load(os.path.join(modelpath, path_) , map_location=torch.device('cpu')))
print(sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
_ = model.to(device)


# ============================================================================================================================================================================
# Starting Evaluation over the validation set
# ============================================================================================================================================================================


history, horizon = 12, 20
unused = cfg['sequence_length_val'] - history - horizon

l2 = torch.nn.MSELoss(reduction = 'none')
loss_fn =  losses.NormalCRPS(reduction="none", mode = cfg['loss_mode'], dim = 1)

print("Adapting lat lon")
print("Lat range: ", cfg['lat_range'])
print("Lon range: ", cfg['lon_range'])
lsm = lsm[cfg['lat_range'][0]:cfg['lat_range'][1], cfg['lon_range'][0]:cfg['lon_range'][1]]


result_list = []

for dataset in datasets:

    if dataset == "oras5":
        val_dl = val_dl_oras5
        val_data = val_data_oras5
    elif dataset == "cesm2_lens":
        val_dl = val_dl_lens
        val_data = val_data_lens
    elif dataset == "cesm2_picontrol":
        val_dl = val_dl_picontrol
        val_data = val_data_picontrol
    elif dataset == "godas":
        val_dl = val_dl_godas
        val_data = val_data_godas

    rmse = 0
    mse = 0
    crps = 0
    nino_rmse_3 = 0
    nino_rmse_3_l = 0
    nino_rmse_34 = 0
    nino_rmse_34_l = 0
    nino_rmse_4 = 0
    nino_rmse_4_l = 0
    rmse_orig = 0
    crps_orig = 0

    print("Starting eval forward pass ViT")
    with torch.no_grad():
        for i, batch in enumerate(val_dl):
            #unpack sample
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            timeseries, context = batch['x'], batch['month'] 
            x, y, _ = timeseries.split([history, horizon, unused], dim = 2)

            # Adapt lat lon range     lat_range: (0, 52),lon_range: (44, 160),
            x = x[:, :, :,
                            cfg['lat_range'][0]:cfg['lat_range'][1],
                            cfg['lon_range'][0]:cfg['lon_range'][1]]
            
            y = y[:, :, :,
                            cfg['lat_range'][0]:cfg['lat_range'][1],
                            cfg['lon_range'][0]:cfg['lon_range'][1]]

            # Forward pass
            prediction = model.forward(x, y, cfg, train=False)

            #get target shape and land-sea fraction
            B, C, T, H, W = y.shape
            N = (B * C* T * ((H * W) - lsm.sum())).item()
            #print("Number of parameters of the model: ", N)
            N = N/len(cfg['in_vars'])       # C
            N = N/horizon                   # T

            if cfg["probabilistic"]:
                # Unpack prediction
                mean_pred, std_pred = prediction[:,0], prediction[:,1]

                # Unnormalize
                mean_pred = einops.rearrange(mean_pred, "b t c h w -> b c t h w")
                std_pred = einops.rearrange(std_pred, "b t c h w -> b c t h w")

                if dataset == "cesm2_lens":
                    mean_pred = normalizer_lens.inverse(mean_pred, device=device)
                    std_pred = normalizer_lens.inverse(std_pred, device=device)
                    target = normalizer_lens.inverse(y, device=device)
                else:
                    mean_pred = val_data.normalizer.inverse(mean_pred, device=device)
                    std_pred = val_data.normalizer.inverse(std_pred, device=device)
                    target = val_data.normalizer.inverse(y, device=device)

                
                std_pred = einops.rearrange(std_pred, "b c t h w -> b t c h w")
                pred = einops.rearrange(mean_pred, "b c t h w -> b t c h w")
                targ = einops.rearrange(target, "b c t h w -> b t c h w")

                std_pred = std_pred[:, :, 0, :, :]
            else:
                # Unnormalize
                prediction = einops.rearrange(prediction, "b t c h w -> b c t h w")
                if dataset == "cesm2_lens":
                    prediction = normalizer_lens.inverse(prediction, device=device)
                    target = normalizer_lens.inverse(y, device=device)
                else:
                    prediction = val_data.normalizer.inverse(prediction, device=device)
                    target = val_data.normalizer.inverse(y, device=device)

                pred = einops.rearrange(prediction, "b c t h w -> b t c h w")
                targ = einops.rearrange(target, "b c t h w -> b t c h w")
            
            pred = pred[:, :, 0, :, :]
            targ = targ[:, :, 0, :, :]

            # Save predictions and targets as zarr file for further analysis
            time_steps = range(pred.shape[0])  # 8 time steps
            lags = range(pred.shape[1])   # 20 variables
            latitude = range(pred.shape[2])    # 52 latitudes
            longitude = range(pred.shape[3])   # 132 longitudes

            pred_path = f'/mnt/qb/goswami/data/data_deeps2a_enso/preds_xr_vit_{model_num}_{dataset}.zarr'
            targ_path = f'/mnt/qb/goswami/data/data_deeps2a_enso/targets_xr_vit_{model_num}_{dataset}.zarr'
            
            # Check if the Zarr file exists
            if os.path.exists(pred_path) and os.path.exists(targ_path):
                # Open the existing datasets
                pred_dataset = xr.open_zarr(pred_path)
                targ_dataset = xr.open_zarr(targ_path)

                # Get the maximum time value in the existing datasets
                max_time = pred_dataset['time'].max().item()
            else:
                # If the Zarr files do not exist, initialize max_time to -1
                max_time = -1

            # Initialize lists to store DataArray objects
            pred_list = []
            targ_list = []
            cfg_ = cfg.copy()
            cfg_.pop('device', None)

            # Iterate over time steps and collect DataArrays in batches of 30
            for t in time_steps:
                # Calculate the new time coordinate
                new_time = max_time + 1 + t

                coords_={
                        "time": [new_time],
                        "lags": lags,
                        "lat": latitude,
                        "lon": longitude
                    }

                if cfg["probabilistic"] == False:
                    pred_ = pred.cpu().numpy()[t][np.newaxis, :, :, :]
                    dims_ = ("time", "lags", "lat", "lon")
                else:
                    pred_ = pred.unsqueeze(1)
                    std_pred_ = std_pred.unsqueeze(1)
                    concat_pred = torch.cat([pred_, std_pred_], dim=1)
                    pred_ = concat_pred.cpu().numpy()[t][np.newaxis, :, :, :, :]
                    dims_ = ("time", "num_pred", "lags", "lat", "lon")
                    coords_["num_pred"] = range(pred_.shape[1])

                # Create the xarray DataArray for predictions
                pred_xr = xr.DataArray(
                    pred_,
                    dims=dims_,
                    coords=coords_,
                    name="predictions_vit",
                    attrs={ "cfg": cfg_,
                            "month": context.cpu().numpy()[t][:history].tolist()}  # Store the cfg dictionary as metadata
                )

                # Create the xarray DataArray for targets
                targ_xr = xr.DataArray(
                    targ.cpu().numpy()[t][np.newaxis, :, :, :],
                    dims=("time", "lags", "lat", "lon"),
                    coords={
                        "time": [new_time],
                        "lags": lags,
                        "lat": latitude,
                        "lon": longitude
                    },
                    name="targets_vit",
                    attrs={ "cfg": cfg_,
                            "month": context.cpu().numpy()[t][history:].tolist()}  # Store the cfg dictionary as metadata
                )

                # Append the DataArrays to their respective lists
                pred_list.append(pred_xr)
                targ_list.append(targ_xr)


            # Concatenate along the 'time' dimension
            pred_concat = xr.concat(pred_list, dim="time").to_dataset(name="predictions_vit")
            targ_concat = xr.concat(targ_list, dim="time").to_dataset(name="targets_vit")

            # Write to Zarr file
            if os.path.exists(pred_path) and os.path.exists(targ_path):
                # Append the concatenated dataset to the existing Zarr files
                pred_concat.to_zarr(pred_path, mode='a', append_dim='time')
                targ_concat.to_zarr(targ_path, mode='a', append_dim='time')
            else:
                # If Zarr files do not exist, create new ones
                pred_concat.to_zarr(pred_path, mode='w')
                targ_concat.to_zarr(targ_path, mode='w')

            #calculate loss
            mse_ = l2(pred, targ)
            mse_ *= (1 - lsm)[None, None, :, :]

            mse += mse_.sum(dim=(0, 2, 3)).div(N)
            rmse += torch.sqrt(mse_.sum(dim=(0, 2, 3)).div(N))
            rmse_orig += torch.sqrt(mse_.mean(dim=0))

            if cfg["probabilistic"]:
                crps_ = loss_fn(targ, torch.stack([pred, std_pred], dim=1))
                crps_ *= (1 - lsm)[None, None, :, :]
                crps += crps_.sum(dim=(0, 2, 3)).div(N)
                crps_orig += crps_.mean(dim=0)


            # Calculate the Nino 3 index - Nino #3 region from 26 to 36 and 80 to 140
            nino_pred_3 = pred[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_3_lon"][0]:cfg["nino_3_lon"][1]]
            nino_true_3 = targ[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_3_lon"][0]:cfg["nino_3_lon"][1]]
            nino_rmse_3 += torch.sqrt(l2(nino_pred_3, nino_true_3).mean(dim=0))
            nino_rmse_3_l += torch.sqrt(l2(nino_pred_3, nino_true_3).mean(dim=(0, 2, 3)))

            # Calculate the Nino3.4 index - Nino3.4 region from 26 to 36 and 60 to 110
            nino_pred_34 = pred[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
            nino_true_34 = targ[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
            nino_rmse_34 += torch.sqrt(l2(nino_pred_34, nino_true_34).mean(dim=0))
            nino_rmse_34_l += torch.sqrt(l2(nino_pred_34, nino_true_34).mean(dim=(0, 2, 3)))

            # Calculate the Nino 4 index - Nino 4 region from 26 to 36 and 30 to 80
            nino_pred_4 = pred[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]
            nino_true_4 = targ[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]
            nino_rmse_4 += torch.sqrt(l2(nino_pred_4, nino_true_4).mean(dim=0))
            nino_rmse_4_l += torch.sqrt(l2(nino_pred_4, nino_true_4).mean(dim=(0, 2, 3)))


        rmse /= len(val_dl)
        mse /= len(val_dl)
        crps /= len(val_dl)
        nino_rmse_3 /= len(val_dl)
        nino_rmse_3_l /= len(val_dl)
        nino_rmse_34 /= len(val_dl)
        nino_rmse_34_l /= len(val_dl)
        nino_rmse_4 /= len(val_dl)
        nino_rmse_4_l /= len(val_dl)
        rmse_orig /= len(val_dl)
        crps_orig /= len(val_dl)

        print("DATASET: ", dataset)
        print("MSE over validation set: ", mse.mean(), mse.shape)
        print("RMSE-div(N)-mean over validation set: ", rmse.mean(), rmse.shape)
        print("RMSE-original over validation set: ", rmse_orig.mean(), rmse_orig.shape)
        print("CRPS over validation set: ", crps.mean(), crps.shape) if cfg["probabilistic"] else None
        print("CRPS_orig over validation set: ", crps_orig.mean(), crps_orig.shape) if cfg["probabilistic"] else None
        print("Nino 3 rmse: ", nino_rmse_3_l.mean(), nino_rmse_3.mean(), nino_rmse_3.shape)
        print("Nino 3.4 rmse: ", nino_rmse_34_l.mean(), nino_rmse_34.mean(), nino_rmse_34.shape)
        print("Nino 4 rmse: ", nino_rmse_4_l.mean(), nino_rmse_4.mean(), nino_rmse_4.shape)

    
    # ============================================================================================================================================================================
    # Save results to file
    # ============================================================================================================================================================================

    results_vit = {
                "dataset" : dataset,
                "model_path" : model_path,
                "rmse" : rmse.cpu(),
                "rmse_orig" : rmse_orig,
                "nino_rmse_3" : nino_rmse_3_l.cpu(),
                "nino_rmse_34" : nino_rmse_34_l.cpu(),
                "nino_rmse_4" : nino_rmse_4_l.cpu(),
                "config": cfg}

    if cfg["probabilistic"]:
        results_vit["crps"] = crps.cpu()
        results_vit["crps_orig"] = crps_orig.cpu()

    result_list.append(results_vit)


# Save results to a pickle file
with open(PATH + "/results/results_vit_"+ str(model_num) + run_specification + ".pkl", 'wb') as file:
    pickle.dump(result_list, file)

