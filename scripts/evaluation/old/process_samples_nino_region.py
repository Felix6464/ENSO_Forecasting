import pickle
import matplotlib.pyplot as plt
import torch
import io, math
import numpy as np
import zarr

from s2aenso.utils import data, normalization, metric

def apply_mask(arr):
    
    nino_mask = np.mean(arr[:, 11], axis=(1, 2)) > 0.5
    nina_mask = np.mean(arr[:, 11], axis=(1, 2)) < -0.5
    neutral_mask = ~nino_mask & ~nina_mask

    idx_nino = np.where(nino_mask)[0].tolist()
    idx_nina = np.where(nina_mask)[0].tolist()
    idx_neutral = np.where(neutral_mask)[0].tolist()

    return idx_nino, idx_nina, idx_neutral


def calculate_3_month_sliding_mean(data_list, nino_index=None):
    # Initialize a list to store the new arrays after calculating the 3-month sliding window mean
    new_data_list = []

    # Iterate over each numpy array in the input list
    for data in data_list:
        # Make a copy of the input data to avoid modifying the original
        pred = np.copy(data[0])
        targ = np.copy(data[1])

        # Apply Nino Index slicing if specified
        if nino_index == "Nino34":
            pred = pred[:, :, 21:31, 42:92]
            targ = targ[:, :, 21:31, 42:92]
        elif nino_index == "Nino4":
            pred = pred[:, :, 21:31, 12:62]
            targ = targ[:, :, 21:31, 12:62]

        # Initialize arrays to store the results
        mean_pred = np.zeros_like(pred)
        mean_targ = np.zeros_like(targ)

        # Iterate over each batch and calculate the 3-month sliding window mean
        for i in range(pred.shape[0]):  # Batch size (8)
            for j in range(pred.shape[1]):  # Lead time (20)
                # Determine the start and end indices for the sliding window
                start_idx = max(0, j - 2)
                end_idx = j + 1

                # Compute the mean over the latitude and longitude dimensions for the sliding window
                mean_pred[i, j] = np.mean(pred[i, start_idx:end_idx, :, :], axis=(0))
                mean_targ[i, j] = np.mean(targ[i, start_idx:end_idx, :, :], axis=(0))

        # Append the processed data to the new data list
        new_data_list.append((mean_pred, mean_targ))

    return new_data_list


model_num_swin = 547314
model_num_vit = 547313
grid_1_1 = False

if grid_1_1:
    nino_3_lat = (21, 31)
    nino_34_lon = (42, 92)
    nino_4_lon = (12, 62)
else:
    nino_3_lat = (21, 31)
    nino_34_lon = (46, 71)
    nino_4_lon = (31, 56)


swinlstm_preds = zarr.open(f'/mnt/qb/goswami/data/data_deeps2a_enso/preds_swinlstm_{model_num_swin}.zarr', mode='r')
swinlstm_targets = zarr.open(f'/mnt/qb/goswami/data/data_deeps2a_enso/targets_swinlstm_{model_num_swin}.zarr', mode='r')
print("SwinLSTM Data loaded")
print(swinlstm_targets.keys())

predictions_swin = swinlstm_preds["predictions"]
context_pred_swin = swinlstm_preds["context"]
targets_swin = swinlstm_targets["targets"]
context_targ_swin = swinlstm_targets["context"]

vit_preds = zarr.open(f'/mnt/qb/goswami/data/data_deeps2a_enso/preds_vit_{model_num_vit}.zarr', mode='r')
vit_targets = zarr.open(f'/mnt/qb/goswami/data/data_deeps2a_enso/targets_vit_{model_num_vit}.zarr', mode='r')
print("ViT Data loaded")
print(vit_targets.keys())

predictions_vit = vit_preds["predictions"]
context_pred_vit = vit_preds["context"]
targets_vit = vit_targets["targets"]
context_targ_vit = vit_targets["context"]


pred_nino_swin = []
targ_nino_swin = []
ctx_pred_nino_swin = []
ctx_targ_nino_swin = []

pred_nina_swin = []
targ_nina_swin = []
ctx_pred_nina_swin = []
ctx_targ_nina_swin = []

pred_neutral_swin = []
targ_neutral_swin = []
ctx_pred_neutral_swin = []
ctx_targ_neutral_swin = []

pred_nino_vit = []
targ_nino_vit = []
ctx_pred_nino_vit = []
ctx_targ_nino_vit = []

pred_nina_vit = []
targ_nina_vit = []
ctx_pred_nina_vit = []
ctx_targ_nina_vit = []

pred_neutral_vit = []
targ_neutral_vit = []
ctx_pred_neutral_vit = []
ctx_targ_neutral_vit = []

for i in range(len(predictions_swin)):
    targ_swin = targets_swin[i][:]
    ctx_targ_swin = context_targ_swin[i][:]
    targ_vit = targets_vit[i][:]
    ctx_targ_vit = context_targ_vit[i][:]

    targ_swin = targ_swin[:, :, nino_3_lat[0]:nino_3_lat[1], nino_34_lon[0]:nino_34_lon[1]]
    idx_nino, idx_nina, idx_neutral = apply_mask(targ_swin)

    pred_swin = predictions_swin[i][:]
    ctx_pred_swin = context_pred_swin[i][:]
    pred_vit = predictions_vit[i][:]
    ctx_pred_vit = context_pred_vit[i][:]

    pred_nino_swin.append(pred_swin[idx_nino, ...])
    ctx_pred_nino_swin.append(ctx_pred_swin[idx_nino, ...])
    targ_nino_swin.append(targ_swin[idx_nino, ...])
    ctx_targ_nino_swin.append(ctx_targ_swin[idx_nino, ...])

    pred_nina_swin.append(pred_swin[idx_nina, ...])
    ctx_pred_nina_swin.append(ctx_pred_swin[idx_nina, ...])
    targ_nina_swin.append(targ_swin[idx_nina, ...])
    ctx_targ_nina_swin.append(ctx_targ_swin[idx_nina, ...])

    pred_neutral_swin.append(pred_swin[idx_neutral, ...])
    ctx_pred_neutral_swin.append(ctx_pred_swin[idx_neutral, ...])
    targ_neutral_swin.append(targ_swin[idx_neutral, ...])
    ctx_targ_neutral_swin.append(ctx_targ_swin[idx_neutral, ...])

    pred_nino_vit.append(pred_vit[idx_nino, ...])
    ctx_pred_nino_vit.append(ctx_pred_vit[idx_nino, ...])
    targ_nino_vit.append(targ_vit[idx_nino, ...])
    ctx_targ_nino_vit.append(ctx_targ_vit[idx_nino, ...])

    pred_nina_vit.append(pred_vit[idx_nina, ...])
    ctx_pred_nina_vit.append(ctx_pred_vit[idx_nina, ...])
    targ_nina_vit.append(targ_vit[idx_nina, ...])
    ctx_targ_nina_vit.append(ctx_targ_vit[idx_nina, ...])

    pred_neutral_vit.append(pred_vit[idx_neutral, ...])
    ctx_pred_neutral_vit.append(ctx_pred_vit[idx_neutral, ...])
    targ_neutral_vit.append(targ_vit[idx_neutral, ...])
    ctx_targ_neutral_vit.append(ctx_targ_vit[idx_neutral, ...])

print("Complete")

zarr_pred_targ_nino_swin = zarr.open("/mnt/qb/goswami/data/data_deeps2a_enso/enso_conditions/swin_pred_targ_nino_"+ str(model_num_swin) + ".zarr", 'w')
zarr_pred_targ_nina_swin = zarr.open("/mnt/qb/goswami/data/data_deeps2a_enso/enso_conditions/swin_pred_targ_nina_"+ str(model_num_swin) + ".zarr", 'w')
zarr_pred_targ_neutral_swin = zarr.open("/mnt/qb/goswami/data/data_deeps2a_enso/enso_conditions/swin_pred_targ_neutral_"+ str(model_num_swin) + ".zarr", 'w')

zarr_pred_targ_nino_vit = zarr.open("/mnt/qb/goswami/data/data_deeps2a_enso/enso_conditions/vit_pred_targ_nino_"+ str(model_num_vit) + ".zarr", 'w')
zarr_pred_targ_nina_vit = zarr.open("/mnt/qb/goswami/data/data_deeps2a_enso/enso_conditions/vit_pred_targ_nina_"+ str(model_num_vit) + ".zarr", 'w')
zarr_pred_targ_neutral_vit = zarr.open("/mnt/qb/goswami/data/data_deeps2a_enso/enso_conditions/vit_pred_targ_neutral_"+ str(model_num_vit) + ".zarr", 'w')

for i, pred in enumerate(pred_nino_swin):
    zarr_pred_targ_nino_swin.create_dataset(f"preds/{i}", data=pred)
    zarr_pred_targ_nino_swin.create_dataset(f"contexts_preds/{i}", data=ctx_pred_nino_swin[i])
    zarr_pred_targ_nino_swin.create_dataset(f"targs/{i}", data=targ_nino_swin[i])
    zarr_pred_targ_nino_swin.create_dataset(f"contexts_targs/{i}", data=ctx_targ_nino_swin[i])

for i, pred in enumerate(pred_nina_swin):
    zarr_pred_targ_nina_swin.create_dataset(f"preds/{i}", data=pred)
    zarr_pred_targ_nina_swin.create_dataset(f"contexts_preds/{i}", data=ctx_pred_nino_swin[i])
    zarr_pred_targ_nina_swin.create_dataset(f"targs/{i}", data=targ_nina_swin[i])
    zarr_pred_targ_nina_swin.create_dataset(f"contexts_targs/{i}", data=ctx_targ_nina_swin[i])

for i, pred in enumerate(pred_neutral_swin):
    zarr_pred_targ_neutral_swin.create_dataset(f"preds/{i}", data=pred)
    zarr_pred_targ_neutral_swin.create_dataset(f"contexts_preds/{i}", data=ctx_pred_nino_swin[i])
    zarr_pred_targ_neutral_swin.create_dataset(f"targs/{i}", data=targ_neutral_swin[i])
    zarr_pred_targ_neutral_swin.create_dataset(f"contexts_targs/{i}", data=ctx_targ_neutral_swin[i])

for i, pred in enumerate(pred_nino_vit):
    zarr_pred_targ_nino_vit.create_dataset(f"preds/{i}", data=pred)
    zarr_pred_targ_nino_vit.create_dataset(f"contexts_preds/{i}", data=ctx_pred_nino_swin[i])
    zarr_pred_targ_nino_vit.create_dataset(f"targs/{i}", data=targ_nino_vit[i])
    zarr_pred_targ_nino_vit.create_dataset(f"contexts_targs/{i}", data=ctx_targ_nino_vit[i])

for i, pred in enumerate(pred_nina_vit):
    zarr_pred_targ_nina_vit.create_dataset(f"preds/{i}", data=pred)
    zarr_pred_targ_nina_vit.create_dataset(f"contexts_preds/{i}", data=ctx_pred_nino_swin[i])
    zarr_pred_targ_nina_vit.create_dataset(f"targs/{i}", data=targ_nina_vit[i])
    zarr_pred_targ_nina_vit.create_dataset(f"contexts_targs/{i}", data=ctx_targ_nina_vit[i])

for i, pred in enumerate(pred_neutral_vit):
    zarr_pred_targ_neutral_vit.create_dataset(f"preds/{i}", data=pred)
    zarr_pred_targ_neutral_vit.create_dataset(f"contexts_preds/{i}", data=ctx_pred_nino_swin[i])
    zarr_pred_targ_neutral_vit.create_dataset(f"targs/{i}", data=targ_neutral_vit[i])
    zarr_pred_targ_neutral_vit.create_dataset(f"contexts_targs/{i}", data=ctx_targ_neutral_vit[i])


'''
print("Start 3 month mean")
print("Pred swin", type(predictions_swin), len(predictions_swin))

nino_index = "Nino4"

roll_mean_pred_targ_swin = calculate_3_month_sliding_mean((predictions_swin, targets_swin), nino_index=nino_index)
roll_mean_pred_targ_vit = calculate_3_month_sliding_mean((predictions_vit, targets_vit), nino_index=nino_index)


zarr_roll_mean_pred_targ_swin = zarr.open(f"/mnt/qb/goswami/data/data_deeps2a_enso/3month_mean/swin_roll_mean_pred_targ_"+ nino_index + "_" + str(model_num_swin) + ".zarr", 'w')
zarr_roll_mean_pred_targ_vit = zarr.open(f"/mnt/qb/goswami/data/data_deeps2a_enso/3month_mean/vit_roll_mean_pred_targ_"+ nino_index + "_" + str(model_num_vit) + ".zarr", 'w')

print("Saving 3 month mean")
print("Example sample: ", roll_mean_pred_targ_swin[0][0].shape, roll_mean_pred_targ_vit[0][1].shape)

for i, roll_mean in enumerate(roll_mean_pred_targ_swin):
    zarr_roll_mean_pred_targ_swin.create_dataset(f"roll_mean/{i}", data=np.array(roll_mean))

for i, roll_mean in enumerate(roll_mean_pred_targ_vit):
    zarr_roll_mean_pred_targ_vit.create_dataset(f"roll_mean/{i}", data=np.array(roll_mean))

'''