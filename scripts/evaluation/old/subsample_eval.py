import pickle
import matplotlib.pyplot as plt
import torch
import io, math, sys
from pathlib import Path
import numpy as np
import zarr
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
from dateutil.relativedelta import relativedelta

sys.path.append(str(Path(__file__).resolve().parents[2]))

from s2aenso.utils import data, normalization, metric
from s2aenso.utils.utilities import *
from s2aenso.model import swinlstm, losses


models_vit = [(765364, "1_2g_schedP_lower_levels"), (765368, "1_2g_schedP"), (765066, "1_1g_schedP"), (767326, "1_2g_oc"), (767345, "7var"),
              (770445, "1_2g_loss_scaling"), (770940, "1var"), (773873, "1var_tauxy"),
              (773919, "3var_tauxy"), (773920, "5var_tauxy"), (778495, "5var"), (783541, "3var"), (783685, "1ens_64_128"),
              (783768, "1ens_128_256"), (783769, "1ens_256_512"), (783771, "1ens_512_1024"), (783793, "10ens_512_1024"),
              (794316, "10ens_128_256"), (794342, "10ens_256_512"), (794353, "10ens_64_128"), (795364, "25ens_64_128"),
              (795376, "25ens_128_256"), (796095, "25ens_256_512"), (802996, "25ens_512_1024"), (803963, "90ens_128_256"),
              (803970, "90ens_256_512"), (808714, "90ens_512_1024")]

models_swin = [(808715, "10ens_2l_256h"), (783794, "10ens_3l_256h"), (795365, "25ens_2l_72h"),
              (805059, "25ens_3l_112h"), (796093, "25ens_2l_256h"), (796157, "25ens_3l_256h"), (808767, "90ens_2l_72h"),
              (802997, "90ens_3l_112h"), (803971, "90ens_3l_256h")]



models_vit = [(783685, "1ens_64_128"), (783768, "1ens_128_256"), (783769, "1ens_256_512"), (783771, "1ens_512_1024"), (783793, "10ens_512_1024"),
              (794316, "10ens_128_256"), (794342, "10ens_256_512"), (794353, "10ens_64_128"), (795364, "25ens_64_128"), (795376, "25ens_128_256"), (796095, "25ens_256_512"), (802996, "25ens_512_1024"), (803963, "90ens_128_256"),
              (803970, "90ens_256_512"), (808714, "90ens_512_1024")]



models_vit = [(765368, "1_2g_schedP", "base"), (765066, "1_1g_schedP", "high_res"), (767326, "1_2g_oc", "oc_lr"), (767345, "7var", "7_level"), (778495, "5var", "5_level"), (783541, "3var", "3_level"), (770940, "1var", "1_level"),
              (773920, "5var_tauxy", "5_level_" + r'$\tau_xy$'), (773919, "3var_tauxy","3_level_" + r'$\tau_xy$'), (773873, "1var_tauxy", "1_level_" + r'$\tau_xy$'),
              (815035, "7var_stauxy", "7_level_" + r'$s\tau_xy$'), (815033, "5var_stauxy", "5_level_" + r'$s\tau_xy$'), (815029, "3var_stauxy", "3_level_" + r'$s\tau_xy$'), (815022, "1var_stauxy", "1_level_" + r'$s\tau_xy$'),
              (765364, "1_2g_schedP_lower_levels", "lower_level"), (815132, "lower_levels_stauxy", "lower_level_" + r'$S\tau_xy$')]





models_swin = [(838612, "s1", "s1"), (838620, "s2", "s2"), (838621, "s3", "s3"), (838624, "s4", "s4"), (838671, "s5", "s5"),
               (838673, "s6", "s6"), (838694, "s7", "s7"), (838698, "s8", "s8"), (838932, "s9", "s9"), (838937, "s10", "s10"),
               (838951, "s11", "s11"), (839072, "s12", "s12"), (839098, "s13", "s13"), (839388, "s14", "s14"), (839472, "no_condition_stauxy", "no_FiLM_stauxy")]

models_vit = [(839158, "s1", "s1"), (839160, "s2", "s2"), (839161, "s3", "s3"), (839164, "s4", "s4"), (839165, "s5", "s5"),
              (839386, "s6", "s6"), (839389, "s7", "s7"), (839396, "s8", "s8"), (839386, "s9", "s9"), (839389, "s10", "s10"),
              (839396, "s11", "s11")]


models_swin = [(838694, "s7", "s7"), (838698, "s8", "s8"), (838932, "s9", "s9"), (838937, "s10", "s10"),
               (838951, "s11", "s11"), (839072, "s12", "s12"), (839098, "s13", "s13"), (839388, "s14", "s14"), (839472, "no_condition_stauxy", "no_FiLM_stauxy")]




models_swin = [(839072, "s12", "s12")]
models_vit = [(815035, "7var_stauxy", "7_level_" + r'$s\tau_xy$')]

num_subsamples = 10

dataset = "oras5" # "cesm2_picontrol" or "oras5" or "cesm2_lens"
model = "swinlstm" # "swinlstm" or "vit"
three_month_mean = True

for model_num_ in models_swin:

    model_num = model_num_[0]
    run_specification = model_num_[1]

    print("MODEL NUMBER : ", model_num)
    print("RUN SPECIFICATION : ", run_specification)

    PATH = f"C:/Users/felix/PycharmProjects/deeps2a-enso/scripts/evaluation/results/unprocessed/raw_preds_targs/{model}_{model_num}_{run_specification}/"
    save_dir_ = f"C:/Users/felix/PycharmProjects/deeps2a-enso/scripts/evaluation/results/processed/{dataset}/processed_results_{model}_{model_num}_{run_specification}/"
    create_directory(save_dir_)


    pred_path = PATH + f"preds_xr_{model}_{model_num}_{dataset}.zarr"
    targ_path = PATH + f"targets_xr_{model}_{model_num}_{dataset}.zarr"


    # ============================================================================================================================
    # Load Data
    # ============================================================================================================================

    # Open the existing datasets
    pred_dataset = xr.open_zarr(pred_path, consolidated=False)
    targ_dataset = xr.open_zarr(targ_path, consolidated=False)

    # Calculate the standard deviation for the datasets
    pred_std = pred_dataset.std(dim='time')
    targ_std = targ_dataset.std(dim='time')

    # Get the first sample from the dataset to extract config
    sample = pred_dataset.isel(time=0)
    pred_data = sample["predictions_vit"] if model == "vit" else sample["predictions_swinlstm"]

    # Access the attributes from the .zattrs file
    zattrs = pred_data.attrs
    cfg = zattrs["cfg"]
    cfg["dataset"] = dataset
    adjust_grid_region = cfg["1_2_grid"]
    temp_ocean_levels = cfg["temp_ocean"]
    cfg["three_month_mean"] = three_month_mean
    cfg["adjust_grid_region"] = adjust_grid_region
    cfg["temp_ocean_levels"] = temp_ocean_levels

    # Get the time range
    total_length = len(pred_dataset['time'])
    num_batches_total = math.ceil(total_length / cfg["batch_size"])

    # Load data and lsm mask
    if cfg["1_2_grid"]:
        val_ds_adapt = xr.open_dataset(PATH + "/../../../../results/val_ds_adapt_1_2_grid.nc")
        common_lsm = xr.open_dataset(PATH + '/../../../../../../data/processed_data/enso_data_pacific/land_sea_mask_common_1_2_grid.nc')['lsm'].data
    else:
        val_ds_adapt = xr.open_dataset(PATH + "/../../../../results/val_ds_adapt.nc")
        common_lsm = xr.open_dataset(PATH + '/../../../../../../data/processed_data/enso_data_pacific/land_sea_mask_common.nc')['lsm'].data

    lsm = common_lsm[cfg["lat_range"][0]:cfg["lat_range"][1], cfg["lon_range"][0]:cfg["lon_range"][1]]

    # Initialize Loss Functions
    l2 = torch.nn.MSELoss(reduction = 'none')
    loss_fn =  losses.NormalCRPS(reduction="none", mode = cfg["loss_mode"], dim = 1)


    # ============================================================================================================================
    # Eval Forward Pass
    # ============================================================================================================================

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

    # Initialize batch-level metrics
    batch_mse = 0
    batch_rmse = 0
    batch_rmse_orig = 0
    batch_crps = 0
    batch_crps_orig = 0
    batch_nino_rmse_3 = 0
    batch_nino_rmse_3_l = 0
    batch_nino_rmse_34 = 0
    batch_nino_rmse_34_l = 0
    batch_nino_rmse_4 = 0
    batch_nino_rmse_4_l = 0

    num_batches = 0

    predictions = []
    targets = []

    # Dictionary to store results for each subsample
    loss_dict = {}

    # Generate evenly spaced indices for subsampling
    subsample_indices = np.linspace(num_batches_total/num_subsamples, num_batches_total-1, num=num_subsamples, dtype=int) * 8
    num_subsamples = len(subsample_indices)
    print("Total length of dataset: ", total_length)
    print("Number of batches: ", num_batches_total)
    print("Number of subsamples: ", num_subsamples)
    print("Subsample indices: ", subsample_indices / 8)

    # Iterate over the dataset in batches of size cfg["batch_size"]
    #for i in range(0, int(total_length/(num_subsamples/4)), cfg["batch_size"]):
    for i in range(0, total_length, cfg["batch_size"]):
        
        batch_end = min(i + cfg["batch_size"], total_length)  # Ensure we don't exceed subsample size

        # Select the corresponding batch of data for the current time steps
        pred_slice = pred_dataset['predictions_vit'].isel(time=slice(i, batch_end)) if model == "vit" else pred_dataset['predictions_swinlstm'].isel(time=slice(i, batch_end))
        target_slice = targ_dataset['targets_vit'].isel(time=slice(i, batch_end)) if model == "vit" else targ_dataset['targets_swinlstm'].isel(time=slice(i, batch_end))

        # Convert the data slices to PyTorch tensors
        pred = torch.tensor(pred_slice.values)  #[8, 20, 51, 120]
        targ = torch.tensor(target_slice.values) #[8, 20, 51, 120]

        # If necessary, adjust grid region
        if adjust_grid_region:
            pred = pred[:, :, :, 29:95] if not cfg["probabilistic"] else pred[:, :, :, :, 29:95]
            targ = targ[:, :, :, 29:95]
            cfg["nino_3_lon"] = (31, 61) 
            cfg["nino_34_lon"] = (21, 46)
            cfg["nino_4_lon"] = (6, 31)
            lsm = common_lsm[cfg["lat_range"][0]:cfg["lat_range"][1], 29:95]

        predictions.append((pred, pred_slice.attrs["month"])) if not cfg["probabilistic"] else predictions.append((pred[:, 0], pred_slice.attrs["month"]))
        targets.append((targ, target_slice.attrs["month"]))

        # Get target shape and land-sea fraction
        B, T, H, W = targ.shape
        N = (B * T * ((H * W) - lsm.sum())).item()
        N = N / T  # Adjust for time

        # Probabilistic case: calculate CRPS if necessary
        if cfg["probabilistic"]:
            crps_ = loss_fn(targ, pred)
            crps_ *= (1 - lsm)[None, None, :, :]
            batch_crps += crps_.sum(dim=(0, 2, 3)).div(N)
            batch_crps_orig += crps_.mean(dim=0)
            pred = pred[:, 0]  # Only use the mean prediction for the deterministic case

        #print("Pred shape: ", pred.shape)
        #print("Targ shape: ", targ.shape)

        # Calculate MSE and RMSE
        mse_ = l2(pred, targ)
        mse_ *= (1 - lsm)[None, None, :, :]
        batch_mse += mse_.sum(dim=(0, 2, 3)).div(N)
        batch_rmse += torch.sqrt(mse_.sum(dim=(0, 2, 3)).div(N))
        batch_rmse_orig += torch.sqrt(mse_.mean(dim=0))
        #print("RMSE orig mean: ", torch.sqrt(mse_.mean(dim=0)).mean())
        #print("RMSE", torch.sqrt(mse_.sum(dim=(0, 2, 3)).div(N)))

        # Calculate Nino 3, 3.4, and 4 indices
        nino_pred_3 = pred[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_3_lon"][0]:cfg["nino_3_lon"][1]]
        nino_true_3 = targ[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_3_lon"][0]:cfg["nino_3_lon"][1]]
        batch_nino_rmse_3 += torch.sqrt(l2(nino_pred_3, nino_true_3).mean(dim=0))
        batch_nino_rmse_3_l += torch.sqrt(l2(nino_pred_3, nino_true_3).mean(dim=(0, 2, 3)))


        nino_pred_34 = pred[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
        nino_true_34 = targ[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
        batch_nino_rmse_34 += torch.sqrt(l2(nino_pred_34, nino_true_34).mean(dim=0))
        batch_nino_rmse_34_l += torch.sqrt(l2(nino_pred_34, nino_true_34).mean(dim=(0, 2, 3)))


        nino_pred_4 = pred[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]
        nino_true_4 = targ[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]
        batch_nino_rmse_4 += torch.sqrt(l2(nino_pred_4, nino_true_4).mean(dim=0))
        batch_nino_rmse_4_l += torch.sqrt(l2(nino_pred_4, nino_true_4).mean(dim=(0, 2, 3)))


        # Increment the batch counter
        num_batches += 1

        if i in subsample_indices:
            # Store the batch results in the dictionary for this subsample iteration
            loss_dict[f'subsample_{i // cfg["batch_size"]}'] = {
                'mse': batch_mse.cpu().numpy() / num_batches,
                'rmse': batch_rmse.cpu().numpy() / num_batches,
                'rmse_orig': batch_rmse_orig.cpu().numpy() / num_batches,
                'crps': batch_crps.cpu().numpy() / num_batches if cfg["probabilistic"] else None,
                'crps_orig': batch_crps_orig.cpu().numpy() / num_batches if cfg["probabilistic"] else None,
                'nino_rmse_3': batch_nino_rmse_3.cpu().numpy() / num_batches,
                'nino_rmse_3_l': batch_nino_rmse_3_l.cpu().numpy() / num_batches,
                'nino_rmse_34': batch_nino_rmse_34.cpu().numpy() / num_batches,
                'nino_rmse_34_l': batch_nino_rmse_34_l.cpu().numpy() / num_batches,
                'nino_rmse_4': batch_nino_rmse_4.cpu().numpy() / num_batches,
                'nino_rmse_4_l': batch_nino_rmse_4_l.cpu().numpy() / num_batches
            }
            print("Subsample num Batches: ", i // cfg["batch_size"])
            print("MSE: ", batch_mse.mean().cpu().numpy() / num_batches)
            print("RMSE: ", batch_rmse.mean().cpu().numpy() / num_batches)
            print("RMSE_orig: ", batch_rmse_orig.mean().cpu().numpy() / num_batches)
            print("CRPS: ", batch_crps.mean().cpu().numpy() / num_batches) if cfg["probabilistic"] else None
            print("CRPS_orig: ", batch_crps_orig.mean().cpu().numpy() / num_batches) if cfg["probabilistic"] else None
            print("Nino 3 rmse: ", batch_nino_rmse_3_l.mean().cpu().numpy() / num_batches)
            print("Nino 3.4 rmse: ", batch_nino_rmse_34_l.mean().cpu().numpy() / num_batches)
            print("Nino 4 rmse: ", batch_nino_rmse_4_l.mean().cpu().numpy() / num_batches)
            print("=====================================================================================================")

            # Accumulate the metrics across all subsamples
            mse += batch_mse / num_batches
            rmse += batch_rmse / num_batches
            rmse_orig += batch_rmse_orig / num_batches
            if cfg["probabilistic"]:
                crps += batch_crps / num_batches
                crps_orig += batch_crps_orig / num_batches
            nino_rmse_3 += batch_nino_rmse_3 / num_batches
            nino_rmse_3_l += batch_nino_rmse_3_l / num_batches
            nino_rmse_34 += batch_nino_rmse_34 / num_batches
            nino_rmse_34_l += batch_nino_rmse_34_l / num_batches
            nino_rmse_4 += batch_nino_rmse_4 / num_batches
            nino_rmse_4_l += batch_nino_rmse_4_l / num_batches

            # Initialize batch-level metrics
            batch_mse = 0
            batch_rmse = 0
            batch_rmse_orig = 0
            batch_crps = 0
            batch_crps_orig = 0
            batch_nino_rmse_3 = 0
            batch_nino_rmse_3_l = 0
            batch_nino_rmse_34 = 0
            batch_nino_rmse_34_l = 0
            batch_nino_rmse_4 = 0
            batch_nino_rmse_4_l = 0

            num_batches = 0



    # After all batches/subsamples, store the summed-up results in the dictionary
    loss_dict['total'] = {
        'mse': mse.cpu().numpy() / num_subsamples,
        'rmse': rmse.cpu().numpy() / num_subsamples,
        'rmse_orig': rmse_orig.cpu().numpy() / num_subsamples,
        'crps': crps.cpu().numpy() / num_subsamples if cfg["probabilistic"] else None,
        'crps_orig': crps_orig.cpu().numpy() / num_subsamples if cfg["probabilistic"] else None,
        'nino_rmse_3': nino_rmse_3.cpu().numpy() / num_subsamples,
        'nino_rmse_3_l': nino_rmse_3_l.cpu().numpy() / num_subsamples,
        'nino_rmse_34': nino_rmse_34.cpu().numpy() / num_subsamples,
        'nino_rmse_34_l': nino_rmse_34_l.cpu().numpy() / num_subsamples,
        'nino_rmse_4': nino_rmse_4.cpu().numpy() / num_subsamples,
        'nino_rmse_4_l': nino_rmse_4_l.cpu().numpy() / num_subsamples,
        'pred_data_std': pred_std,
        'targ_data_std': targ_std
    }

    rmse /= num_subsamples
    mse /= num_subsamples
    crps /= num_subsamples
    nino_rmse_3 /= num_subsamples
    nino_rmse_3_l /= num_subsamples
    nino_rmse_34 /= num_subsamples
    nino_rmse_34_l /= num_subsamples
    nino_rmse_4 /= num_subsamples
    nino_rmse_4_l /= num_subsamples
    rmse_orig /= num_subsamples
    crps_orig /= num_subsamples

    print("MSE over validation set: ", mse.mean(), mse.shape)
    print("RMSE-div(N)-mean over validation set: ", rmse.mean(), rmse.shape)
    print("RMSE-original over validation set: ", rmse_orig.mean(), rmse_orig.shape)
    print("CRPS over validation set: ", crps.mean(), crps.shape) if cfg["probabilistic"] else None
    print("CRPS_orig over validation set: ", crps_orig.mean(), crps_orig.shape) if cfg["probabilistic"] else None
    print("Nino 3 rmse: ", nino_rmse_3_l.mean(), nino_rmse_3.mean(), nino_rmse_3.shape)
    print("Nino 3.4 rmse: ", nino_rmse_34_l.mean(), nino_rmse_34.mean(), nino_rmse_34.shape)
    print("Nino 4 rmse: ", nino_rmse_4_l.mean(), nino_rmse_4.mean(), nino_rmse_4.shape)


    # ============================================================================================================================
    # Calculate Climatology
    # ============================================================================================================================

    if dataset == "oras5":
        val_range = slice('1983-01-16', "2022-09-16")
    elif dataset == "cesm2_picontrol":
        val_range = slice('1700-01-01', "2022-09-16")
    elif dataset == "cesm2_lens":
        val_range = slice('1850-01-01', "2022-09-16")
    else:
        val_range = slice(None, None)

    # Check if ocean temperature is given in levels
    if temp_ocean_levels:

        if cfg["1_2_grid"]:
            if dataset == "cesm2_picontrol":
                ds_sst_level0 = xr.open_dataset("C:/Users/felix/PycharmProjects/deeps2a-enso/data/test_data/cesm2_picontrol_temp_ocean_1_2_grid_combined_level0.nc")
            elif dataset == "oras5":
                ds_sst_level0 = xr.open_dataset("C:/Users/felix/PycharmProjects/deeps2a-enso/data/test_data/temp_ocn_0a_lat-31_33_lon90_330_gr1.0_2.0_level0.nc")
            else:
                ds_sst_level0 = xr.open_dataset("C:/Users/felix/PycharmProjects/deeps2a-enso/data/test_data/ensemble_1301_001_temp_ocn_0a_lat-31_33_lon90_330_gr1.0_2.0_1850-2015.nc")
        else:
            if dataset == "cesm2_picontrol":
                ds_sst_level0 = xr.open_dataset("C:/Users/felix/PycharmProjects/deeps2a-enso/data/test_data/cesm2_picontrol_temp_ocean_combined_level0.nc")
            elif dataset == "oras5":
                ds_sst_level0 = xr.open_dataset("C:/Users/felix/PycharmProjects/deeps2a-enso/data/test_data/temp_ocn_0a_lat-31_33_lon130_290_gr1.0_1.0_level0.nc")
            else:
                ds_sst_level0 = xr.open_dataset("C:/Users/felix/PycharmProjects/deeps2a-enso/data/test_data/ensemble_1301_001_temp_ocn_0a_lat-31_33_lon130_290_gr1.0_1.0_1850-2015.nc")

        # Calculate climatology -> data already in anomalies
        ds_sst_level0 = ds_sst_level0.sel(time=val_range)
        ds_sst_level0_nino3 = ds_sst_level0.sel(lat=slice(-4, 5), lon=slice(210, 269))
        ds_sst_level0_nino34 = ds_sst_level0.sel(lat=slice(-4, 5), lon=slice(190, 239))
        ds_sst_level0_nino4 = ds_sst_level0.sel(lat=slice(-4, 5), lon=slice(174, 223))

        rmse_sst_level0_spatial = np.sqrt(((ds_sst_level0['temp_ocn_0a']**2).mean('time')))
        rmse_sst_level0 = np.sqrt((ds_sst_level0['temp_ocn_0a']**2).mean(dim=['time', 'lat', 'lon']))
        rmse_sst_level0_nino3 = np.sqrt((ds_sst_level0_nino3['temp_ocn_0a']**2).mean(dim=['time', 'lat', 'lon']))
        rmse_sst_level0_nino34 = np.sqrt((ds_sst_level0_nino34['temp_ocn_0a']**2).mean(dim=['time', 'lat', 'lon']))
        rmse_sst_level0_nino4 = np.sqrt((ds_sst_level0_nino4['temp_ocn_0a']**2).mean(dim=['time', 'lat', 'lon']))

        print("RMSE Climatology Mean",rmse_sst_level0.mean())
        print("RMSE Climatology Spatial", rmse_sst_level0_spatial.mean())

        # if not anomalies -> rmse_ssh = np.sqrt(((ds_oras5_ssh.groupby('time.month') - da_oras5_ssh_climatology)**2).mean('time'))

    else:
        if cfg["dataset"] == "cesm2_picontrol":
            ds_sst = xr.open_dataset("C:/Users/felix/PycharmProjects/deeps2a-enso/data/test_data/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssta_lat-31_33_lon130_290_gr1.0.nc")
        elif cfg["dataset"] == "oras5":
            ds_sst = xr.open_dataset("C:/Users/felix/PycharmProjects/deeps2a-enso/data/test_data/ssta_lat-31_33_lon130_290_gr1.0.nc")
        else:
            ds_sst = xr.open_dataset("C:/Users/felix/PycharmProjects/deeps2a-enso/data/test_data/b.e21.BHISTcmip6.f09_g17.LE2-1301.001.ssta_lat-31_33_lon130_290_gr1.0.nc")

        # Calculate climatology -> data already in anomalies
        # Calculate climatology -> data already in anomalies
        ds_sst_level0 = ds_sst.sel(time=val_range)
        ds_sst_level0_nino3 = ds_sst_level0.sel(lat=slice(-4, 5), lon=slice(210, 269))
        ds_sst_level0_nino34 = ds_sst_level0.sel(lat=slice(-4, 5), lon=slice(190, 239))
        ds_sst_level0_nino4 = ds_sst_level0.sel(lat=slice(-4, 5), lon=slice(174, 223))

        rmse_sst_level0_spatial = np.sqrt(((ds_sst_level0['ssta']**2).mean('time')))
        rmse_sst_level0 = np.sqrt((ds_sst_level0['ssta']**2).mean(dim=['time', 'lat', 'lon']))
        rmse_sst_level0_nino3 = np.sqrt((ds_sst_level0_nino3['ssta']**2).mean(dim=['time', 'lat', 'lon']))
        rmse_sst_level0_nino34 = np.sqrt((ds_sst_level0_nino34['ssta']**2).mean(dim=['time', 'lat', 'lon']))
        rmse_sst_level0_nino4 = np.sqrt((ds_sst_level0_nino4['ssta']**2).mean(dim=['time', 'lat', 'lon']))



        # ============================================================================================================================
    # Generat Hindcast Plots
    # ============================================================================================================================


    create_directory(save_dir_ + "/hindcasts/")

    lags = [3, 12, 20]
    val_ds_adapt = val_ds_adapt.sel(lat=slice(-25, 25))

    if dataset == "cesm2_picontrol":
        choose_samples = [10, 25, 50] #[500, 1000, 1500] #total samples 297
    else:
        choose_samples = [10, 25, 45] #total samples 53

    for sample in choose_samples:

        pred, _ = predictions[sample]
        targ, _ = targets[sample]

        if model == "swinlstm":
            pred = pred[:, :, 0:51, :]
            targ = targ[:, :, 0:51, :]
            lsm = lsm[0:51, :]

        if adjust_grid_region:
            val_ds_adapt = val_ds_adapt.sel(lat=slice(-26, 25), lon=slice(148, 279))

                
        for lag in lags:

            hindcast = dict()
            for key, x in dict(Prediction=pred.unsqueeze(1), Target=targ.unsqueeze(1)).items():

                x_lst = []
                x = x.cpu()
                for i, var in enumerate(val_ds_adapt.data_vars):
                    da = xr.DataArray(
                        data= x[:, i, lag - 1],
                        coords=val_ds_adapt.isel(time=[1, 2, 3, 4, 5, 6, 7, 8]).coords,
                        name = var)

                    # Mask land
                    da = da.where(lsm==0, other=np.nan)

                    # Add to list
                    x_lst.append(da)
                    break
                hindcast[key] = xr.merge(x_lst)

            # Create a figure with 2 subplots sharing the same colorbar and projection
            fig, axs = plt.subplots(1, 2, figsize=(11, 5), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180), 'aspect': 'auto'})

            # Define a list of keys, titles, and colorbar parameters
            keys = ['Prediction', 'Target']
            titles = ['Prediction', 'Target']
            cmap = 'RdBu_r'
            vmin = -3
            vmax = 3

            # Flatten the axes array for easier iteration
            axs = axs.ravel()

            for ax, key, title in zip(axs, keys, titles):
                x = hindcast[key]
                for var in x.data_vars:
                    im = ax.pcolormesh(x.lon, x.lat, x[var].isel(time=0), cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
                    ax.coastlines()
                    ax.set_title(title, fontsize=16, pad=15)  # Adjusted padding for subplot titles
                    ax.add_feature(cfeature.BORDERS, linestyle=':')
                    ax.add_feature(cfeature.LAND, facecolor='lightgray')
                    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
                    gl.top_labels = False  # Remove x-labels from the top
                    gl.right_labels = False  # Remove y-labels from the right
                    gl.xlabel_style = {'fontsize': 14}  # Increase x-label font size
                    gl.ylabel_style = {'fontsize': 14}  # Increase y-label font size
                    if ax == axs[1]:  # For the second subplot (right), remove y-labels
                        gl.left_labels = False  # Remove y-labels from the left side of the second plot

                    # Add contour lines
                    ax.contour(x.lon, x.lat, x[var].isel(time=0), levels=2, colors='k', linewidths=0.3, transform=ccrs.PlateCarree())

            # Add a shared colorbar below the plots
            cbar_ax = fig.add_axes([0.2, 0.05, 0.65, 0.04])  # Decreased the space between colorbar and plots
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
            cbar.set_label('SSTA [°C]', fontsize=16)
            cbar.ax.tick_params(labelsize=16)

            # Add a main title for the entire figure
            fig.suptitle(f'Hindcast Comparison at Lag {lag}', fontsize=17, fontweight='bold', y=0.92)  # Adjusted to bring suptitle closer to plots

            # Improve layout, reduce space around plots and suptitle
            plt.tight_layout(rect=[0, 0.1, 1, 0.93])
            #plt.show()
            fig.savefig(f'{save_dir_}/hindcasts/hindcast_comparison_no{sample}_{model}_lag_{lag}.png')
            plt.close()

    # ============================================================================================================================
    # Calculate Skillscore
    # ============================================================================================================================


    # New dictionary to store skill scores for each subsample and total results
    skillscore_dict = {}

    # Initialize variables to accumulate the total skill scores across all subsamples
    total_skillscore_mean = []
    total_skillscore_mean_3i = []
    total_skillscore_mean_34i = []
    total_skillscore_mean_4i = []


    # Iterate over each subsample stored in the results_dict
    for subsample_key, subsample_results in loss_dict.items():

        if subsample_key == 'total':
            continue

        # Initialize lists to store skill scores for the current subsample
        skillscore_mean = []
        skillscore_ = []
        skillscore_mean_3i = []
        skillscore_3i = []
        skillscore_mean_34i = []
        skillscore_34i = []
        skillscore_mean_4i = []
        skillscore_4i = []

        # Calculate skillscore for the entire region
        loss_spatial = subsample_results['rmse_orig'][:, 0:51, :]  # Use the RMSE values for this subsample
        loss_climatology_spatial = rmse_sst_level0_spatial.sel(lat=slice(-25, 25), lon=slice(148, 279))  # Climatology data for skill score calculation
        loss_climatology = rmse_sst_level0.item()

        for lag in range(20):
            rmse_lag_spatial = xr.DataArray(loss_spatial[lag], dims=['lat', 'lon'], coords={'lat': val_ds_adapt.lat, 'lon': val_ds_adapt.lon})
            rmse_lag = loss_spatial[lag].mean()
            skillscore = 1 - (rmse_lag / loss_climatology)
            skillscore_spatial = 1 - (rmse_lag_spatial / loss_climatology_spatial)
            skillscore_.append(skillscore_spatial)
            skillscore_mean.append(skillscore.item())

        # Calculate skillscore for the Nino 3 region
        loss_spatial = subsample_results['nino_rmse_3'][:, 0:51, :]
        val_ds_3i = val_ds_adapt.sel(lat=slice(-4, 5), lon=slice(210, 269))
        loss_climatology_3i_spatial = rmse_sst_level0_spatial.sel(lat=slice(-4, 5), lon=slice(210, 269))
        loss_climatology_3i = rmse_sst_level0_nino3.item()

        for lag in range(20):
            rmse_lag_spatial = xr.DataArray(loss_spatial[lag], dims=['lat', 'lon'], coords={'lat': val_ds_3i.lat, 'lon': val_ds_3i.lon})
            rmse_lag = loss_spatial[lag].mean()
            skillscore = 1 - (rmse_lag / loss_climatology_3i)
            skillscore_spatial = 1 - (rmse_lag_spatial / loss_climatology_3i_spatial)
            skillscore_3i.append(skillscore_spatial)
            skillscore_mean_3i.append(skillscore.item())

        # Calculate skillscore for the Nino 3.4 region
        loss_spatial = subsample_results['nino_rmse_34'][:, 0:51, :]
        val_ds_34i = val_ds_adapt.sel(lat=slice(-4, 5), lon=slice(190, 239))
        loss_climatology_34i_spatial = rmse_sst_level0_spatial.sel(lat=slice(-4, 5), lon=slice(190, 239))
        loss_climatology_34i = rmse_sst_level0_nino34.item()

        for lag in range(20):
            rmse_lag_spatial = xr.DataArray(loss_spatial[lag], dims=['lat', 'lon'], coords={'lat': val_ds_34i.lat, 'lon': val_ds_34i.lon})
            rmse_lag = loss_spatial[lag].mean()
            skillscore = 1 - (rmse_lag / loss_climatology_34i)
            skillscore_spatial = 1 - (rmse_lag_spatial / loss_climatology_34i_spatial)
            skillscore_34i.append(skillscore_spatial)
            skillscore_mean_34i.append(skillscore.item())

        # Calculate skillscore for the Nino 4 region
        loss_spatial = subsample_results['nino_rmse_4'][:, 0:51, :]
        val_ds_4i = val_ds_adapt.sel(lat=slice(-4, 5), lon=slice(174, 223))
        loss_climatology_4i_spatial = rmse_sst_level0_spatial.sel(lat=slice(-4, 5), lon=slice(174, 223))
        loss_climatology_4i = rmse_sst_level0_nino4.item()

        for lag in range(20):
            rmse_lag_spatial = xr.DataArray(loss_spatial[lag], dims=['lat', 'lon'], coords={'lat': val_ds_4i.lat, 'lon': val_ds_4i.lon})
            rmse_lag = loss_spatial[lag].mean()
            skillscore = 1 - (rmse_lag / loss_climatology_4i)
            skillscore_spatial = 1 - (rmse_lag_spatial / loss_climatology_4i_spatial)
            skillscore_4i.append(skillscore_spatial)
            skillscore_mean_4i.append(skillscore.item())

        print(f"Skillscore for subsample {subsample_key}:")
        print("Mean Skillscore: ", skillscore_mean)
        print("Mean Skillscore 3i: ", skillscore_mean_3i)
        print("Mean Skillscore 34i: ", skillscore_mean_34i)
        print("Mean Skillscore 4i: ", skillscore_mean_4i)
        print("=====================================================================================================")

        # Store the skill scores for this subsample in the skillscore_dict
        skillscore_dict[subsample_key] = {
            'skillscore_mean': skillscore_mean,
            'skillscore_': skillscore_,
            'skillscore_mean_3i': skillscore_mean_3i,
            'skillscore_3i': skillscore_3i,
            'skillscore_mean_34i': skillscore_mean_34i,
            'skillscore_34i': skillscore_34i,
            'skillscore_mean_4i': skillscore_mean_4i,
            'skillscore_4i': skillscore_4i
        }

        if not total_skillscore_mean:
            total_skillscore_mean = skillscore_mean
            total_skillscore_ = skillscore_
            total_skillscore_mean_3i = skillscore_mean_3i
            total_skillscore_3i = skillscore_3i
            total_skillscore_mean_34i = skillscore_mean_34i
            total_skillscore_34i = skillscore_34i
            total_skillscore_mean_4i = skillscore_mean_4i
            total_skillscore_4i = skillscore_4i
        else:
            # Accumulate skill scores for calculating total results
            total_skillscore_mean = [a + b for a, b in zip(total_skillscore_mean, skillscore_mean)]
            total_skillscore_ = [a + b for a, b in zip(total_skillscore_, skillscore_)]
            total_skillscore_mean_3i = [a + b for a, b in zip(total_skillscore_mean_3i, skillscore_mean_3i)]
            total_skillscore_3i = [a + b for a, b in zip(total_skillscore_3i, skillscore_3i)]
            total_skillscore_mean_34i = [a + b for a, b in zip(total_skillscore_mean_34i, skillscore_mean_34i)]
            total_skillscore_34i = [a + b for a, b in zip(total_skillscore_34i, skillscore_34i)]
            total_skillscore_mean_4i = [a + b for a, b in zip(total_skillscore_mean_4i, skillscore_mean_4i)]
            total_skillscore_4i = [a + b for a, b in zip(total_skillscore_4i, skillscore_4i)]

    total_skillscore_mean = [a / num_subsamples for a in total_skillscore_mean]
    total_skillscore_ = [a / num_subsamples for a in total_skillscore_]
    total_skillscore_mean_3i = [a / num_subsamples for a in total_skillscore_mean_3i]
    total_skillscore_3i = [a / num_subsamples for a in total_skillscore_3i]
    total_skillscore_mean_34i = [a / num_subsamples for a in total_skillscore_mean_34i]
    total_skillscore_34i = [a / num_subsamples for a in total_skillscore_34i]
    total_skillscore_mean_4i = [a / num_subsamples for a in total_skillscore_mean_4i]
    total_skillscore_4i = [a / num_subsamples for a in total_skillscore_4i]

    # After all subsamples, calculate the total skill scores across all subsamples
    skillscore_dict['total'] = {
        'skillscore_mean': total_skillscore_mean,
        'skillscore_': total_skillscore_,
        'skillscore_mean_3i': total_skillscore_mean_3i,
        'skillscore_3i': total_skillscore_3i,
        'skillscore_mean_34i': total_skillscore_mean_34i,
        'skillscore_34i': total_skillscore_34i,
        'skillscore_mean_4i': total_skillscore_mean_4i,
        'skillscore_4i': total_skillscore_4i
    }

    # Print the final total skill scores
    print("Final Total Skill Scores:")
    for metric_, value in skillscore_dict['total'].items():
        if 'mean' in metric_:
            print(f"{metric_}: {value}")


    # ============================================================================================================================
    # Calculate Monthly Anomaly Correlation Coefficient
    # ============================================================================================================================

    # ============================================================================================================================
    # Calculate Monthly Anomaly Correlation Coefficient
    # ============================================================================================================================

    # ============================================================================================================================
    # Calculate Monthly Anomaly Correlation Coefficient
    # ============================================================================================================================

    pred_dataset_ = pred_dataset["predictions_vit"] if model == "vit" else pred_dataset["predictions_swinlstm"]
    targ_dataset_ = targ_dataset["targets_vit"] if model == "vit" else targ_dataset["targets_swinlstm"]

    if dataset == "cesm2_picontrol":
        if len(pred_dataset['time']) == 15576:
            # Slice the datasets to the possible time range 1700-2000 if whole dataset is given
            pred_dataset_ = pred_dataset_.isel(time=slice(12000, 15576))
            targ_dataset_ = targ_dataset_.isel(time=slice(12000, 15576))
        start_year = 1700
    elif dataset == "oras5":
        start_year = 1983
    else:
        start_year = 1850

    n_time_steps = pred_dataset_.time.size

    start_date = datetime.datetime(start_year, 1, 1)
    date_range_targ = [start_date + relativedelta(months=i) for i in range(n_time_steps)]
    date_range_pred = [start_date + relativedelta(months=i) for i in range(n_time_steps)]

    # Assign the generated date range to the 'time' coordinate of the DataArray
    targ_dataset_ = targ_dataset_.assign_coords(time=date_range_targ)
    pred_dataset_ = pred_dataset_.assign_coords(time=date_range_pred)

    pred_months = pred_dataset_.groupby('time.month')
    targ_months = targ_dataset_.groupby('time.month')

    # Initialize a dictionary to store pred-targ pairs for each month
    monthly_data = {}

    # Iterate over both groups simultaneously using zip
    for (pred_month, pred_group), (targ_month, targ_group) in zip(pred_months, targ_months):
        # Check if the months are aligned
        assert pred_month == targ_month, f"Mismatch in months: {pred_month} != {targ_month}"

        # Convert the xarray.DataArray to NumPy arrays and then to PyTorch tensors
        pred_tensor = torch.tensor(pred_group.values)
        targ_tensor = torch.tensor(targ_group.values)

        pred_tensor = pred_tensor[:, 0,] if cfg["probabilistic"] else pred_tensor

        # Adjust the grid region if necessary
        if adjust_grid_region:
            pred_tensor = pred_tensor[:, :, :, 29:95]
            targ_tensor = targ_tensor[:, :, :, 29:95]
        
        # Save the pred and targ data in the dictionary under the respective month
        monthly_data[pred_month] = {
            'pred': [(pred_tensor, None)],
            'targ': [(targ_tensor, None)]
        }


    monthly_acc = {}

    for month in monthly_data.keys():

        predictions = monthly_data[month]['pred']
        targets = monthly_data[month]['targ']

        #print("predictions", predictions[0].shape)

        results_month = {}

        # Initialize lists to store ACC values for this subsample
        acc_mean = []
        acc_mean_spatial = []
        acc_mean_3i = []
        acc_mean_3i_spatial = []
        acc_mean_34i = []
        acc_mean_34i_spatial = []
        acc_mean_4i = []
        acc_mean_4i_spatial = []

        # Iterate over lags (assuming 20 lags)
        for lag in range(20):
            # Calculate ACC for the entire region
            acc = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg)
            acc_mean.append(np.mean(acc))
            
            acc_spatial = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, keep_spatial_coords=True)
            stacked_np = np.stack(acc_spatial)
            mean_acc_spatial = np.mean(stacked_np, axis=0)
            acc_mean_spatial.append(mean_acc_spatial)

            # Calculate ACC for Nino 3 region
            acc_3i = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, nino_index="nino_3")
            acc_mean_3i.append(np.mean(acc_3i))

            acc_3i_spatial = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, nino_index="nino_3", keep_spatial_coords=True)
            stacked_3i_np = np.stack(acc_3i_spatial)
            mean_acc_3i_spatial = np.mean(stacked_3i_np, axis=0)
            acc_mean_3i_spatial.append(mean_acc_3i_spatial)

            # Calculate ACC for Nino 3.4 region
            acc_34i = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, nino_index="nino_34")
            acc_mean_34i.append(np.mean(acc_34i))

            acc_34i_spatial = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, nino_index="nino_34", keep_spatial_coords=True)
            stacked_34i_np = np.stack(acc_34i_spatial)
            mean_acc_34i_spatial = np.mean(stacked_34i_np, axis=0)
            acc_mean_34i_spatial.append(mean_acc_34i_spatial)

            # Calculate ACC for Nino 4 region
            acc_4i = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, nino_index="nino_4")
            acc_mean_4i.append(np.mean(acc_4i))

            acc_4i_spatial = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, nino_index="nino_4", keep_spatial_coords=True)
            stacked_4i_np = np.stack(acc_4i_spatial)
            mean_acc_4i_spatial = np.mean(stacked_4i_np, axis=0)
            acc_mean_4i_spatial.append(mean_acc_4i_spatial)

        print(f"ACC for month {month}:")
        print("Mean ACC: ", acc_mean)
        print("Mean ACC Nino 3: ", acc_mean_3i)
        print("Mean ACC Nino 3.4: ", acc_mean_34i)
        print("Mean ACC Nino 4: ", acc_mean_4i)

        results_month["acc_mean"] = acc_mean
        results_month["acc_mean_3i"] = acc_mean_3i
        results_month["acc_mean_34i"] = acc_mean_34i
        results_month["acc_mean_4i"] = acc_mean_4i
        results_month["acc_mean_spatial"] = acc_mean_spatial
        results_month["acc_mean_3i_spatial"] = acc_mean_3i_spatial
        results_month["acc_mean_34i_spatial"] = acc_mean_34i_spatial
        results_month["acc_mean_4i_spatial"] = acc_mean_4i_spatial

        monthly_acc[month] = results_month


    # ============================================================================================================================
    # Calculate 3-month rolling mean ACC
    # ============================================================================================================================

    if three_month_mean:
        print("Applying 3-month rolling mean to target values...")

        window_size = 3  # 3-month rolling window

        # Apply rolling window mean over the 'time' dimension
        target_slice = targ_dataset_.isel(lags=0)

        # Select the last time step and take all entries of the 'lags' dimension
        last_time_step_lags = targ_dataset_.isel(time=-1).isel(lags=slice(0, None))

        # Get the last value of the 'time' dimension from the target_slice
        last_time_value = target_slice.time.values[-1].astype('datetime64[D]')

        # Create new 'time' values for the new DataArray from last_time_step_lags
        new_time_values = [last_time_value + np.timedelta64(i, 'D') for i in range(1, last_time_step_lags.lags.size + 1)]

        # Create a new DataArray from last_time_step_lags, with 'time' coordinates extended
        new_dataarray = xr.DataArray(
            data=last_time_step_lags.values,  # Use the values from last_time_step_lags
            dims=target_slice.dims,  # Same dimensions as target_slice
            coords={  # Coordinate system: 'time', 'lat', and 'lon'
                'time': new_time_values,
                'lat': target_slice.lat,
                'lon': target_slice.lon
            }
        )
        # Concatenate the new DataArray with the original target_slice along the 'time' dimension
        target_slice = xr.concat([target_slice, new_dataarray], dim='time')

        # Compute the rolling mean with a window of 3 along the 'time' dimension
        # 'center=True' ensures the window is centered at the current time step
        target_slice_rolling = target_slice.rolling(time=window_size, center=True).mean()

        # Now handle the edge cases (first and last entries)
        # The first entry should remain the same as the original value since there's no previous data
        target_slice_rolling = target_slice_rolling.copy()  # Make a copy to safely modify the data
        target_slice_rolling.isel(time=0).data = target_slice.isel(time=0).data  # Use original value for first entry

        # The last entry should also remain the same as the original value since there's no subsequent data
        target_slice_rolling.isel(time=-1).data = target_slice.isel(time=-1).data  # Use original value for last entry

        print("3-month rolling mean targets have been created...")

    # Initialize lists to store predictions and targets at every iteration
    predictions = []
    targets = []

    # Dictionary to store the predictions and targets for each subsample
    subsample_dict = {}

    # Get the time range
    total_length = len(pred_dataset_.time)
    num_batches_total = math.ceil(total_length / cfg["batch_size"])
    subsample_indices = np.linspace(num_batches_total/num_subsamples, num_batches_total-1, num=num_subsamples, dtype=int) * 8

    # Iterate over the dataset for all time steps
    for i in range(0, len(pred_dataset_.time)):
        
        # Select the corresponding slices from the prediction and target datasets for the current time step
        pred_slice = pred_dataset_.isel(time=i)
        targ_slice = target_slice_rolling.isel(time=slice(i, i + 20))

        # Convert the data slices to PyTorch tensors and add a batch dimension
        pred = torch.tensor(pred_slice.values).unsqueeze(0)  # [1, 20, 51, 120]
        targ = torch.tensor(targ_slice.values).unsqueeze(0)  # [1, 20, 51, 120]

        # Adjust the grid region if necessary
        if adjust_grid_region:
            pred = pred[:, :, :, 29:95] if not cfg["probabilistic"] else pred[:, :, :, :, 29:95]
            targ = targ[:, :, :, 29:95]

        # Store the current predictions and targets in their respective lists
        if not cfg["probabilistic"]:
            predictions.append((pred, pred_slice.attrs["month"]))
        else:
            predictions.append((pred[:, 0], pred_slice.attrs["month"]))
        
        targets.append((targ, target_slice.attrs["month"]))

        # When the current time step 'i' is part of the subsample_indices
        if i in subsample_indices:
            if i == subsample_indices[0]:
                del predictions[0]
                del targets[0]
            # Save the current predictions and targets into the dictionary
            subsample_key = f'subsample_{i}'  # Create a unique key for each subsample
            subsample_dict[subsample_key] = {
                'predictions': predictions.copy(),  # Copy the list to avoid clearing issues
                'targets': targets.copy()  # Copy the list to avoid clearing issues
            }

            # Clear the predictions and targets lists for the next subsample


    # ============================================================================================================================
    # Calculate 3-month rolling mean ACC
    # ============================================================================================================================


    # New dictionary to store the ACC results for each subsample
    subsample_dict_acc = {}

    # Initialize lists to accumulate ACC values over all subsamples
    total_acc_mean = []
    total_acc_mean_spatial = []
    total_acc_mean_3i = []
    total_acc_mean_3i_spatial = []
    total_acc_mean_34i = []
    total_acc_mean_34i_spatial = []
    total_acc_mean_4i = []
    total_acc_mean_4i_spatial = []

    # Initialize counters for lags to accumulate over subsamples
    num_subsamples = len(subsample_dict)

    # Iterate over each subsample in subsample_dict
    for subsample_key, subsample_data in subsample_dict.items():
        
        # Retrieve predictions and targets from the subsample
        predictions = subsample_data['predictions']
        targets = subsample_data['targets']

        # Initialize lists to store ACC values for this subsample
        acc_mean = []
        acc_mean_spatial = []
        acc_mean_3i = []
        acc_mean_3i_spatial = []
        acc_mean_34i = []
        acc_mean_34i_spatial = []
        acc_mean_4i = []
        acc_mean_4i_spatial = []

        # Iterate over lags (assuming 20 lags)
        for lag in range(20):
            # Calculate ACC for the entire region
            acc = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg)
            acc_mean.append(np.mean(acc))
            
            acc_spatial = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, keep_spatial_coords=True)
            stacked_np = np.stack(acc_spatial)
            mean_acc_spatial = np.mean(stacked_np, axis=0)
            acc_mean_spatial.append(mean_acc_spatial)

            # Calculate ACC for Nino 3 region
            acc_3i = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, nino_index="nino_3")
            acc_mean_3i.append(np.mean(acc_3i))

            acc_3i_spatial = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, nino_index="nino_3", keep_spatial_coords=True)
            stacked_3i_np = np.stack(acc_3i_spatial)
            mean_acc_3i_spatial = np.mean(stacked_3i_np, axis=0)
            acc_mean_3i_spatial.append(mean_acc_3i_spatial)

            # Calculate ACC for Nino 3.4 region
            acc_34i = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, nino_index="nino_34")
            acc_mean_34i.append(np.mean(acc_34i))

            acc_34i_spatial = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, nino_index="nino_34", keep_spatial_coords=True)
            stacked_34i_np = np.stack(acc_34i_spatial)
            mean_acc_34i_spatial = np.mean(stacked_34i_np, axis=0)
            acc_mean_34i_spatial.append(mean_acc_34i_spatial)

            # Calculate ACC for Nino 4 region
            acc_4i = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, nino_index="nino_4")
            acc_mean_4i.append(np.mean(acc_4i))

            acc_4i_spatial = metric.anomaly_correlation_coefficient(predictions, targets, lag, cfg, nino_index="nino_4", keep_spatial_coords=True)
            stacked_4i_np = np.stack(acc_4i_spatial)
            mean_acc_4i_spatial = np.mean(stacked_4i_np, axis=0)
            acc_mean_4i_spatial.append(mean_acc_4i_spatial)

        # Save the ACC values for this subsample into the dictionary
        subsample_dict_acc[subsample_key] = {
            'acc_mean': acc_mean,
            'acc_mean_spatial': acc_mean_spatial,
            'acc_mean_3i': acc_mean_3i,
            'acc_mean_3i_spatial': acc_mean_3i_spatial,
            'acc_mean_34i': acc_mean_34i,
            'acc_mean_34i_spatial': acc_mean_34i_spatial,
            'acc_mean_4i': acc_mean_4i,
            'acc_mean_4i_spatial': acc_mean_4i_spatial
        }

        # Print the ACC values for this subsample
        print(f"ACC values for subsample {subsample_key}:")
        print("Mean ACC: ", acc_mean)
        print("Mean ACC 3i: ", acc_mean_3i)
        print("Mean ACC 34i: ", acc_mean_34i)
        print("Mean ACC 4i: ", acc_mean_4i)
        print("=====================================================================================================")

        # Accumulate the results for total ACC calculation
        if not total_acc_mean:
            total_acc_mean = acc_mean
            total_acc_mean_spatial = acc_mean_spatial
            total_acc_mean_3i = acc_mean_3i
            total_acc_mean_3i_spatial = acc_mean_3i_spatial
            total_acc_mean_34i = acc_mean_34i
            total_acc_mean_34i_spatial = acc_mean_34i_spatial
            total_acc_mean_4i = acc_mean_4i
            total_acc_mean_4i_spatial = acc_mean_4i_spatial
        else:
            total_acc_mean = [x + y for x, y in zip(total_acc_mean, acc_mean)]
            total_acc_mean_spatial = [x + y for x, y in zip(total_acc_mean_spatial, acc_mean_spatial)]
            total_acc_mean_3i = [x + y for x, y in zip(total_acc_mean_3i, acc_mean_3i)]
            total_acc_mean_3i_spatial = [x + y for x, y in zip(total_acc_mean_3i_spatial, acc_mean_3i_spatial)]
            total_acc_mean_34i = [x + y for x, y in zip(total_acc_mean_34i, acc_mean_34i)]
            total_acc_mean_34i_spatial = [x + y for x, y in zip(total_acc_mean_34i_spatial, acc_mean_34i_spatial)]
            total_acc_mean_4i = [x + y for x, y in zip(total_acc_mean_4i, acc_mean_4i)]
            total_acc_mean_4i_spatial = [x + y for x, y in zip(total_acc_mean_4i_spatial, acc_mean_4i_spatial)]

    # Calculate the average total ACC over all subsamples
    total_acc_mean = [x / num_subsamples for x in total_acc_mean]
    total_acc_mean_spatial = [x / num_subsamples for x in total_acc_mean_spatial]
    total_acc_mean_3i = [x / num_subsamples for x in total_acc_mean_3i]
    total_acc_mean_3i_spatial = [x / num_subsamples for x in total_acc_mean_3i_spatial]
    total_acc_mean_34i = [x / num_subsamples for x in total_acc_mean_34i]
    total_acc_mean_34i_spatial = [x / num_subsamples for x in total_acc_mean_34i_spatial]
    total_acc_mean_4i = [x / num_subsamples for x in total_acc_mean_4i]
    total_acc_mean_4i_spatial = [x / num_subsamples for x in total_acc_mean_4i_spatial]

    # Save the total ACC values into the dictionary
    subsample_dict_acc['total'] = {
        'acc_mean': total_acc_mean,
        'acc_mean_spatial': total_acc_mean_spatial,
        'acc_mean_3i': total_acc_mean_3i,
        'acc_mean_3i_spatial': total_acc_mean_3i_spatial,
        'acc_mean_34i': total_acc_mean_34i,
        'acc_mean_34i_spatial': total_acc_mean_34i_spatial,
        'acc_mean_4i': total_acc_mean_4i,
        'acc_mean_4i_spatial': total_acc_mean_4i_spatial
    }

    # Print the final total skill scores
    print("Final Total ACC values:")
    for metric_, value in subsample_dict_acc['total'].items():
        if 'spatial' not in metric_:
            print(f"{metric_}: {value}")


    # ============================================================================================================================
    # Save results
    # ============================================================================================================================


    results = {
            "dataset" : dataset,
            "skillscore" : skillscore_dict,
            "acc" : subsample_dict_acc,
            "monthly_acc" : monthly_acc,
            "loss" : loss_dict,
            "config": cfg}


    # Save results to a pickle file
    with open(f"C:/Users/felix/PycharmProjects/deeps2a-enso/scripts/evaluation/results/unprocessed/{dataset}/results_{model}_"+ str(model_num) + "_" + run_specification + ".pkl", 'wb') as file:
        pickle.dump(results, file)