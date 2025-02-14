# %%
import os, time, argparse, wandb
import xarray as xr
import torch 
import einops
import math
from torch import nn
from importlib import reload
from matplotlib import pyplot as plt
import torch.profiler

from s2aenso.distributed.utils import *
from s2aenso.model import geoformer, losses 
from s2aenso.utils import data, normalization, metric, utilities

PATH = os.path.dirname(os.path.abspath(__file__))
id = int(os.environ.get('SLURM_LOCALID', 0))
device= torch.device("cuda", id ) if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device: {device}', flush=True)
torch.cuda.empty_cache()

utilities.set_seed(42)

# ===========================================================================================================================
# Parameters fixed
# ===========================================================================================================================

slurm_id = os.environ.get('SLURM_JOB_ID', 0000000)
cfg = {
    "name": f"vit_cmip6_{slurm_id}",
    "data_dir_zhou" : PATH + '/../../data/processed_data/zhou_data/CMIP6_separate_model_up150m_tauxy_Nor_kb.nc',
    #"data_dir_zhou_val" : PATH + '/../../data/processed_data/zhou_data/SODA_ORAS_group_temp_tauxy_before1979_kb.nc',
    "data_dir_zhou_val" : PATH + '/../../data/processed_data/zhou_data/val_data_subset.nc',
    "lsm_path": PATH + "/../../scripts/data_processing/lsm_zhou.nc", 
    "model_dir" : PATH + "/../../models/vit_zhou/",
    "decay" : False,
    "scheduler_paper": True,
    "one_cycle_lr": False,
    "needtauxy": True,
    "probabilistic": False,
    #training parameters
    "variable_loss_scaling": [1, 1, 1, 1, 1, 1, 1, 0.01, 0.01],
    "cutout" : 0.0,
    "batch_size" : 8,           # Change for testing
    "epochs" : 40,              # Change for testing
    "base_lr" : 6e-4,
    "min_lr": 1e-6,
    "pct_start": (5000/220000),
    "div_factor": 1e10,
    "final_div_factor": 1,
    "weight_decay" : 5e-2,
    "decay_rate" : 0.65,
    "dropout" : 0.2,
    "warmup" : 2000,
    "sv_ratio" : 1,
    #model parameters
    "sequence_length_train" : 32,
    "sequence_length_val" : 32,
    "input_length": 12,
    "output_length": 20,
    "patch_size" : (3, 4),
    "loss_mode" : "sample",
    "d_size": 256,
    "nheads": 4,
    "dim_feedforward": 512,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    #less important
    "device": device,
    "nino_loss": True,
    "gradscaler": False,
    "loss_mean_variant": False,
    #data parameters
    "in_vars" : ["temperatureNor", "tauxNor", "tauyNor"],
    "out_vars" : ["temperatureNor", "tauxNor", "tauyNor"],
    "input_channel": 7,
    "output_channel": 7,
    "num_workers" : 8,
    "lat_range": (0, 51),
    "lon_range": (45, 165),
    "lat_range_relative": (15, 35),
    "lon_range_relative": (49, 75),
    "H0": int((63 - 0) / 3), # lat_range[1] - lat_range[0] / patch_size[0] #(51-0) originally
    "W0": int((160- 0) / 4), # lon_range[1] - lon_range[0] / patch_size[1] #(165-45) originially
    "emb_spatial_size": int((51) / 3) * int((120) / 4),  # H0 * W0 -> 510
    # logging parameters
    "use_wandb" : True,        # Change for testing
    "wb_project" : 'SwinLSTM_Paper',
    "wb_tags": ['single_gpu'],
}


cfg["nino_3_lat"] = (16, 36)
cfg["nino_3_lon"] = (60, 90) 
cfg["nino_34_lon"] = (50, 75)
cfg["nino_4_lon"] = (35, 60)


cfg["ninoweight"] = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6)
            * np.log(np.arange(24) + 1)
        ).to(cfg["device"])
cfg["ninoweight"] = cfg["ninoweight"][: cfg["output_length"]]

# ===========================================================================================================================
# Parameters as Input
# ===========================================================================================================================

config_parsed = {}

parser = argparse.ArgumentParser()
parser.add_argument('-scheduler_paper', '--scheduler_paper', type=bool, help='Whether to us the scheduler from the paper.')
parser.add_argument('-probabilistic', '--probabilistic', type=bool, help='Whether to use ORAS5 data for validation.')
parser.add_argument('-decay', '--decay', type=bool, help='Whether to use ORAS5 data for validation.')
parser.add_argument('-epochs', '--epochs', type=int, help='Whether to use ORAS5 data for validation.')

config_parsed = vars(parser.parse_args())

cfg.update({k: v for k, v in config_parsed.items() if v is not None})

print("CONFIG", cfg)
# ===========================================================================================================================
# Load data
# ===========================================================================================================================

# CESM2
train_data = data.load_zhou_dataset(data_path=cfg['data_dir_zhou'],
                                      vars=cfg['in_vars'],
                                      sequence_length=cfg['sequence_length_train'])

print("LENGTH TRAIN DATA", len(train_data))

train_dl = torch.utils.data.DataLoader(train_data,
                                       batch_size=cfg['batch_size'],
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=True,
                                       num_workers=cfg['num_workers'])

print("LENGTH TRAIN DL", len(train_dl))


val_data  = data.load_zhou_dataset_val(data_path=cfg['data_dir_zhou_val'],
                                      vars=cfg['in_vars'],
                                      sequence_length=cfg['sequence_length_train'])

val_dl = torch.utils.data.DataLoader(val_data,
                                       batch_size=cfg['batch_size'],
                                       shuffle=False,
                                       drop_last=True,
                                       pin_memory=True,
                                       num_workers=cfg['num_workers'])

print("LENGTH VAL DL", len(val_dl))

# Add common land sea mask
common_lsm = xr.open_dataset(cfg['lsm_path'])['lsm'].data
lsm = torch.as_tensor(common_lsm, device = device, dtype = torch.long)

# ===========================================================================================================================
# %%
# Define model
# ===========================================================================================================================

if cfg['probabilistic']:
    from s2aenso.model import geoformer_probabilistic as geoformer
    cfg["loss_mode"] = "parametric"
model = geoformer.Geoformer(config=cfg).to(device)
print("Number of Parameters of Model: ", utilities.count_parameters(model))

# ===========================================================================================================================
# %%
# Loss
# ===========================================================================================================================

decay = [cfg['decay_rate']**i for i in range(cfg['sequence_length_train'])] if cfg['decay_rate'] else [1 for _ in range(cfg['sequence_length_train'])]
loss_decay = torch.as_tensor(decay, device = device, dtype = torch.float)
smooth_l1 = torch.nn.SmoothL1Loss(reduction = 'none')
l2 = torch.nn.MSELoss(reduction = 'none')

loss_fn =  losses.NormalCRPS(reduction="none", mode = cfg['loss_mode'], dim = 1)

# ===========================================================================================================================
# %%
# Optimizer and Scheduler
# ===========================================================================================================================
gradscaler = torch.cuda.amp.GradScaler()
num_iter_per_epoch, num_epochs = len(train_dl), cfg['epochs']


if cfg['scheduler_paper']:
    adam = torch.optim.Adam(model.parameters(), lr=0)
    factor = math.sqrt(cfg['d_size'] * cfg["warmup"]) * 0.0015
    optim = losses.lrwarm(cfg['d_size'], factor, cfg["warmup"], optimizer=adam)
elif cfg['one_cycle_lr']:
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['base_lr'],
                          weight_decay=cfg['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=optim,
                    max_lr=cfg['base_lr'],
                    epochs=num_epochs,
                    steps_per_epoch=num_iter_per_epoch,
                    pct_start=cfg['pct_start'],
                    anneal_strategy='cos',
                    div_factor=cfg["div_factor"],
                    final_div_factor=cfg["final_div_factor"])
else:
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['base_lr'],
                          weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optim, T_max=num_epochs*num_iter_per_epoch, eta_min=cfg['min_lr'])

# ===========================================================================================================================
# %%
# Create model directory
# ===========================================================================================================================

model_dir = os.path.join(cfg['model_dir'], cfg['name'])
ckpt_path = os.path.join(model_dir, 'cpkt.pth') 
best_ckpt_path = os.path.join(model_dir, 'best.pth')

# Check if model exists
if os.path.exists(ckpt_path):
    raise ValueError('Model exists! Terminate script.')

# Create directory
if not os.path.exists(model_dir):
    print(f"Create directoty {model_dir}", flush=True)
    os.makedirs(model_dir)

# Save config
with open(os.path.join(model_dir, "config.json"), 'w') as file:
    json.dump(cfg, file, cls=ExtendedJSONEncoder, indent=4)

# Setup wandb
if cfg['use_wandb']:
    os.environ['WANDB__SERVICE_WAIT'] = '1200' 
    exp_name = cfg['name'] 
    wandb.init(
        project=cfg['wb_project'],
        name=exp_name,
        dir=model_dir,
        config=cfg,
        entity='climate_ml'
    )

# ===========================================================================================================================
# %%
# Forward Pass
# ===========================================================================================================================

def forward_step(batch, cfg: dict, lsm, model, loss_fn, loss_decay, l2, 
                mode: str='train', arch: str='vit'):
    """
    Performs a forward pass and returns the loss.
    """
    #unpack sample
    batch = {k: v.to(cfg["device"], non_blocking=True) for k, v in batch.items()}
    timeseries, context = batch['x'], batch['month'] 

    # Randomly split the timeseries into history and horizon
    #history = torch.randint(1, config['max_history'], (1,))
    history = cfg['input_length']
    horizon = cfg['output_length']
    predictor, predictand = timeseries.split([history, horizon], dim = 2)
    
    # Select lat lon range
    predictor = predictor[:, :, :,
                    cfg['lat_range'][0]:cfg['lat_range'][1],
                    cfg['lon_range'][0]:cfg['lon_range'][1]]
    
    predictand = predictand[:, :, :,
                    cfg['lat_range'][0]:cfg['lat_range'][1],
                    cfg['lon_range'][0]:cfg['lon_range'][1]]
    # Predictor shape: [8, 7, 12, 64, 160]
    # Predictand shape: [8, 7, 20, 64, 160]
        
    lsm = lsm[cfg['lat_range'][0]:cfg['lat_range'][1], cfg['lon_range'][0]:cfg['lon_range'][1]]

    # Supervised ratio
    if cfg["sv_ratio"] > 0:
        cfg["sv_ratio"] = max(cfg['sv_ratio'] - 2.5e-4, 0)

    # Forward pass
    train = True if mode == 'train' else False
    prediction = model.forward(predictor, predictand, cfg, train=train, sv_ratio=cfg["sv_ratio"])
    # Prediction shape: [8, 20, 7, 64, 160] train || [8, 20, 7, 64, 160] val
    if arch == "vit":
        predictand = einops.rearrange(predictand, 'B C T H W -> B T C H W')
        # Predictand shape: [8, 20, 7, 64, 160]


    #get target shape and land-sea fraction
    B, T, C, H, W = predictand.shape
    N = (B * C * T * ((H * W) - lsm.sum())).item()
    
    # Calculate Nino Loss
    if cfg["nino_loss"]:
        nino_pred = prediction[:, 0, :, 0, :, :] if cfg["probabilistic"] else prediction[:, :, 0, :, :]
        nino_pred = nino_pred[
                                :,
                                :,
                                cfg["lat_range_relative"][0] : cfg["lat_range_relative"][1],
                                cfg["lon_range_relative"][0] : cfg["lon_range_relative"][1],
                                ]
        nino_pred = nino_pred.mean(dim = (2, 3))    

        nino_true = predictand[:, :, 0, :, :]
        nino_true = nino_true[
                                :,
                                :,
                                cfg["lat_range_relative"][0] : cfg["lat_range_relative"][1],
                                cfg["lon_range_relative"][0] : cfg["lon_range_relative"][1],
                            ]
        nino_true = nino_true.mean(dim = (2, 3))
        loss_nino = losses.loss_nino(nino_pred, nino_true.float())
        nino_score = losses.cal_ninoscore(cfg["ninoweight"], nino_pred, nino_true.float())
        
    # Select only level 0 of sst for validation
    if mode == "val":
        prediction = prediction[:, :, :, 0, :, :] if cfg["probabilistic"] else prediction[:, :, 0, :, :]
        predictand = predictand[:, :, 0, :, :]
        N = N / C # since only level 0 of sst is used for loss

        if cfg["probabilistic"]:
            # Calculate Nino Index
            nino_pred_34 = prediction[:, :, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
            nino_true_34 = predictand[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]

            nino_pred_4 = prediction[:, :, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]
            nino_true_4 = predictand[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]

            # calculate ACC
            acc_mean = metric.anomaly_correlation_coefficient([(prediction[:, 0], None)], [(predictand, None)], None, None, None)
            acc_mean_34 = metric.anomaly_correlation_coefficient([(nino_pred_34[:, 0], None)], [(nino_true_34, None)], None, None, None)
            acc_mean_4 = metric.anomaly_correlation_coefficient([(nino_pred_4[:, 0], None)], [(nino_true_4, None)], None, None, None)

        else:
            # Calculate Nino Index
            nino_pred_34 = prediction[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
            nino_true_34 = predictand[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]

            nino_pred_4 = prediction[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]
            nino_true_4 = predictand[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]

            # calculate ACC
            acc_mean = metric.anomaly_correlation_coefficient([(prediction, None)], [(predictand, None)], None, None, None)
            acc_mean_34 = metric.anomaly_correlation_coefficient([(nino_pred_34, None)], [(nino_true_34, None)], None, None, None)
            acc_mean_4 = metric.anomaly_correlation_coefficient([(nino_pred_4, None)], [(nino_true_4, None)], None, None, None)

        crps_34 = loss_fn(observation = nino_true_34, prediction = nino_pred_34)
        crps_4 = loss_fn(observation = nino_true_4, prediction = nino_pred_4)
        mse_34 = l2(nino_pred_34[:, 0], nino_true_34) if cfg["probabilistic"] else l2(nino_pred_34, nino_true_34)
        mse_4 = l2(nino_pred_4[:, 0], nino_true_4) if cfg["probabilistic"] else l2(nino_pred_4, nino_true_4)

    #calculate loss
    crps = loss_fn(observation = predictand, prediction = prediction)
    mse = l2(prediction[:, 0], predictand) if cfg["probabilistic"] else l2(prediction, predictand)


    if cfg["nino_loss"]:
        mse_nino = l2(nino_pred, nino_true)

    if len(crps.shape) == 4:
        crps = crps.unsqueeze(0)
        mse = mse.unsqueeze(0)   
        if cfg["nino_loss"]:
            mse_nino = mse_nino.unsqueeze(0)

    # Apply land-sea mask
    #set loss to 0 for land pixels
    crps *= (1 - lsm)[None, None, None, :, :]
    mse *= (1 - lsm)[None, None, None, :, :]

    if mode == "train":
        # Define weights for each variable in the loss
        variable_weights = torch.tensor(cfg["variable_loss_scaling"], device=device).view(1, 1, -1, 1, 1)
        
        # Apply weights to the loss
        crps *= variable_weights
        mse *= variable_weights

    if cfg["decay"]:
        #calculate discounted loss
        if mode == 'train':
            crps = crps.sum(dim = (0, 2, 3, 4)) * loss_decay[:T]
            mse = mse.sum(dim = (0, 2, 3, 4)) * loss_decay[:T]

    #calculate mean loss
    if cfg["loss_mean_variant"]:
        avg_crps = crps.mean()
        avg_rmse = mse.mean().sqrt()
        avg_mse = mse.mean()
    else:
        avg_crps = crps.sum().div(N)
        avg_rmse = mse.sum().div(N).sqrt()
        avg_mse = mse.sum().div(N)

        if mode == "val":
            avg_crps_34 = crps_34.sum().div(N)
            avg_crps_4 = crps_4.sum().div(N)
            avg_rmse_34 = mse_34.sum().div(N).sqrt()
            avg_rmse_4 = mse_4.sum().div(N).sqrt()

    if cfg["nino_loss"]:
        avg_rmse_nino = mse_nino.mean().sqrt()

    
    if cfg["probabilistic"]:
        loss = avg_crps
        if cfg["nino_loss"]:
            #loss_nino = losses.log_scale_loss(loss_nino)
            loss_nino = avg_rmse_nino / 10
            loss = loss + loss_nino.to(cfg["device"])
    else:
        loss = avg_rmse
        if cfg["nino_loss"]:
            #loss_nino = losses.log_scale_loss(loss_nino)
            loss_nino = avg_rmse_nino / 10
            loss = loss + loss_nino.to(cfg["device"])

    print("LOSS:", loss.item())

    #log metrics
    logger = {}
    logger["mse_avg"] = avg_mse.item()
    logger['rmse_avg'] = avg_rmse.item()
    logger['crps_avg'] = avg_crps.item()
    logger['loss_avg'] = loss.item()

    if cfg["nino_loss"]:
        logger['nino_loss'] = loss_nino.item()
        logger['nino_score'] = nino_score
    if mode == "val":
        logger["acc_mean"] = acc_mean.item()
        logger["acc_mean_34"] = acc_mean_34.item()
        logger["acc_mean_4"] = acc_mean_4.item()
        logger['crps_avg_34'] = avg_crps_34.item()
        logger['crps_avg_4'] = avg_crps_4.item()
        logger['rmse_avg_34'] = avg_rmse_34.item()
        logger['rmse_avg_4'] = avg_rmse_4.item()

    
    return loss, logger

# ===========================================================================================================================
# %%
# Main training loop 
# ===========================================================================================================================

print(f"Training model {cfg['name']}", flush=True)

#Training helper variables
val_loss = 1e5
val_loss_oras5 = 1e5
val_loss_picontrol = 1e5
best_val_loss = 1e4
best_val_loss_oras5 = 1e4
best_val_loss_picontrol = 1e4

metric_tracker = []

for current_epoch in range(num_epochs):
    epoch_tracker = {}    
    start_time = time.time()

    
    print("Starting Val Loop")
    model.eval()
    mode = 'val'
    if cfg["nino_loss"]:
        metric_logger = {f"rmse_avg": 0.0, f"rmse_avg_34": 0.0, f"rmse_avg_4": 0.0, f"crps_avg": 0.0, f"crps_avg_34": 0.0, f"crps_avg_4": 0.0, f"loss_avg":0.0, f"mse_avg": 0.0, f"nino_score": 0.0, f"nino_loss": 0.0, f"acc_mean": 0.0}
    else:
        metric_logger = {f"rmse_avg": 0.0, f"rmse_avg_34": 0.0, f"rmse_avg_4": 0.0, f"crps_avg": 0.0, f"crps_avg_34": 0.0, f"crps_avg_4": 0.0, f"loss_avg":0.0, f"mse_avg": 0.0, f"acc_mean": 0.0}


    with torch.no_grad():
        for i, batch in enumerate(val_dl):
            val_loss, logger = forward_step(batch, cfg, lsm, model, 
                                            loss_fn, loss_decay, l2, mode=mode)
            #print("Loss val: ", val_loss.item())
            for key in metric_logger.keys():
                    metric_logger[key] += logger[key]

        for key in metric_logger.keys():
            epoch_tracker[f"{mode}/{key}"] = metric_logger[key] / len(val_dl)
            print(f"{mode}/{key}: ", epoch_tracker[f"{mode}/{key}"])


    #Train loop
    print("Starting Train Loop")
    model.train()
    mode = 'train'
    if cfg["nino_loss"]:
        metric_logger = {f"rmse_avg": 0.0, f"crps_avg":0.0, f"loss_avg":0.0, f"mse_avg": 0.0, f"nino_score": 0.0, f"nino_loss": 0.0}
    else:
        metric_logger = {f"rmse_avg": 0.0, f"crps_avg":0.0, f"loss_avg":0.0, f"mse_avg": 0.0}
    
    for i, batch in enumerate(train_dl):

        if cfg["scheduler_paper"]: 
            optim.optimizer.zero_grad()
        else: 
            optim.zero_grad()


        with torch.cuda.amp.autocast(enabled=cfg["gradscaler"]):

            loss, logger = forward_step(batch, cfg, lsm, model, 
                        loss_fn, loss_decay, l2, mode=mode)
            #print("Loss train: ", loss.item())
            
            if cfg["gradscaler"]:
                gradscaler.scale(loss).backward()
            else:
                loss.backward()

            if cfg["scheduler_paper"]:
                optim.step()
            else:
                if cfg["gradscaler"]:
                    gradscaler.step(optim)
                    gradscaler.update()
                    scheduler.step()
                else:
                    optim.step()
                    scheduler.step()
            

            for key in metric_logger.keys():
                metric_logger[key] += logger[key]

    for key in metric_logger.keys():
        epoch_tracker[f"{mode}/{key}"] = metric_logger[key] / len(train_dl)
        print(f"{mode}/{key}: ", epoch_tracker[f"{mode}/{key}"])


    #Print progress
    print(f'Epoch: {current_epoch}, Train Loss: {epoch_tracker["train/loss_avg"]:,.2e}'
          + f'Val Loss: {epoch_tracker["val/loss_avg"]:,.2e},' 
          + f'Val RMSE: {epoch_tracker["val/rmse_avg"]:,.2e}', flush=True)
    
    print(f'Elapsed time:  {(time.time() - start_time)/60 :.2f} minutes', flush=True)

    # Wandb
    if cfg['use_wandb']:
        wandb.log(epoch_tracker)
    
    # Checkpoint
    torch.save(model.state_dict(), ckpt_path)

    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print("Saving new best model")
        torch.save(model.state_dict(), best_ckpt_path)

    # Store metrics
    metric_tracker.append(epoch_tracker)
    buff = {key:[] for key in epoch_tracker.keys()}
    for d in metric_tracker:
        for key, val in d.items():
            buff[key].append(val)
    with open(os.path.join(model_dir, "metrics.json"), 'w') as file:
        json.dump(buff, file, cls=ExtendedJSONEncoder, indent=4)