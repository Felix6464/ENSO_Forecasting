
# %%
import os, time, argparse, wandb
import xarray as xr
import torch 
from torch import nn
from importlib import reload
from matplotlib import pyplot as plt

from s2aenso.distributed.utils import *
from s2aenso.model import swinlstm, losses 
from s2aenso.utils import data, normalization

PATH = os.path.dirname(os.path.abspath(__file__))
id = int(os.environ.get('SLURM_LOCALID', 0))
device= torch.device("cuda", id ) if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device: {device}', flush=True)

# Parameters
# ======================================================================================
slurm_id = os.environ.get('SLURM_JOB_ID', 0000000)
cfg = {
    "name": f"base_swinlstm_cera20c_parametric_{slurm_id}",
    "lens_dir" : PATH + '/../../data/processed_data/cesm2_lens/historical/',
    "cera20c_dir" : PATH + '/../../data/processed_data/oras5/',
    "oras5_dir" : PATH + '/../../data/processed_data/cera-20c/',
    "lsm_path": PATH + "/../../data/processed_data/land_sea_mask_common.nc", 
    "model_dir" : PATH + "/../../models/basemodel_cera20c/",
    #data parameters
    "in_vars" : ['ssta', 'ssha'], #ssta ssha tauxa
    "out_vars" : ['ssta', 'ssha'], #ssta ssha tauxa
    "sequence_length" : 28,
    "max_history" : 4,
    "train_horizon": 16,
    "num_workers" : 2,
    #training parameters
    "batch_size" : 4, 
    "base_batch_size": 8, # Only required for CosineAnnealing
    "epochs" : 50,
    "base_lr" : 1e-3,
    "min_lr": 1e-6,
    "pct_start": 0.2,
    "base_beta1" : 0.9,
    "base_beta2" : 0.995,
    "weight_decay" : 5e-2,
    "decay_rate" : 0.65,
    "use_checkpointing" : True,
    "mixed_precision" : True,
    "scale_lr": False,
    "scale_beta1": False,
    "scale_beta2": False,
    "seed" : None,
    "clip_gradients" : 5.,
    "scheduler_step": "batch", # When update step takes place
    "val_loss_name": "loss",
    "test_loss_name": "rmse",
    #model parameters
    "num_channels" : 128,
    "num_layers" : 2,
    "num_conditions" : 12,
    "num_tails" : 2,
    "patch_size" : (4, 4),
    "k_conv" : 7,
    "loss_mode" : "parametric",
    # logging parameters
    "use_wandb" : True,
    "wb_project" : 'S2A_Basemodels',
    "wb_tags": ['single_gpu'],
}

# %%
# Load data
# ======================================================================================
reload(data)
reload(normalization)
normalizer = normalization.Normalizer_LENS(cfg['in_vars'])
# CERA20C
train_data = data.load_reanalysis(root_dir = cfg['cera20c_dir'],
                                  vars = cfg['in_vars'], 
                                  sequence_length= cfg['sequence_length'],
                                  normalizer=normalizer,
                                  split= 'train')

train_dl = torch.utils.data.DataLoader(train_data, 
                                       batch_size=cfg['batch_size'], 
                                       shuffle=True, 
                                       drop_last=True,
                                       pin_memory=True,
                                       num_workers=cfg['num_workers'])

# CERA20C 
val_data = data.load_reanalysis(root_dir = cfg['cera20c_dir'],
                                  vars = cfg['in_vars'], 
                                  sequence_length= cfg['sequence_length'],
                                  normalizer=normalizer,
                                  split= 'val')
val_dl = torch.utils.data.DataLoader(val_data, 
                                  batch_size=cfg['batch_size'], 
                                  shuffle=False, 
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=cfg['num_workers'])
# ORAS5 
val2_data = data.load_reanalysis(root_dir = cfg['oras5_dir'],
                                  vars = cfg['in_vars'], 
                                  sequence_length = cfg['sequence_length'],
                                  normalizer=normalizer,
                                  split= 'train')

val2_dl = torch.utils.data.DataLoader(val2_data, 
                                  batch_size=cfg['batch_size'], 
                                  shuffle=False, 
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=cfg['num_workers'])

# Add common land sea mask
common_lsm = xr.open_dataset(cfg['lsm_path'])['lsm'].data
lsm = torch.as_tensor(common_lsm, device = device, dtype = torch.long)

# %%
# Define model
# ======================================================================================
model = swinlstm.SwinLSTMNet(
    input_dim = len(cfg['in_vars']), 
    output_dim= len(cfg['out_vars']),
    num_channels=cfg['num_channels'],
    num_layers= cfg['num_layers'],
    patch_size=cfg['patch_size'],
    num_tails= cfg['num_tails'],
    k_conv=cfg['k_conv'],
    num_conditions= cfg['num_conditions']
).to(device)

# %%
# Loss
# ======================================================================================
decay = [cfg['decay_rate']**i for i in range(cfg['sequence_length'])] if cfg['decay_rate'] else [1 for _ in range(cfg['sequence_length'])]
loss_decay = torch.as_tensor(decay, device = device, dtype = torch.float)
smooth_l1 = torch.nn.SmoothL1Loss(reduction = 'none')
l2 = torch.nn.MSELoss(reduction = 'none')

loss_fn =  losses.NormalCRPS(reduction="none", mode = cfg['loss_mode'], dim = 1)

# %%
# Optimizer and Scheduler
# ======================================================================================
gradscaler = torch.cuda.amp.GradScaler()
optim = torch.optim.AdamW(model.parameters(), lr=cfg['base_lr'],
                          weight_decay=cfg['weight_decay'])
num_iter_per_epoch, num_epochs = len(train_dl), cfg['epochs']
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optim, T_max=num_epochs*num_iter_per_epoch, eta_min=cfg['min_lr']
)
# %%
# Create model directory
# ======================================================================================
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
    os.environ['WANDB__SERVICE_WAIT'] = '600' 
    exp_name = cfg['name'] 
    wandb.init(
        project=cfg['wb_project'],
        name=exp_name,
        dir=model_dir,
        config=cfg,
    )
# %%
def forward_step(batch, cfg: dict, dataset, lsm, model, loss_fn, loss_decay, l2, 
                 mode: str='train'):
    """
    Performs a forward pass and returns the loss.
    """
    #unpack sample
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    timeseries, context = batch['x'], batch['month'] 
    # Randomly split the timeseries into history and horizon
    #history = torch.randint(1, cfg['max_history'], (1,))
    #horizon = cfg['sequence_length'] - history
    history = cfg['max_history'] 
    horizon = cfg['train_horizon'] if mode == 'train' else cfg['sequence_length'] - history 
    unused = cfg['sequence_length'] - history - horizon
    x, y, _ = timeseries.split([history, horizon, unused], dim = 2)

    # Select output variables
    idx_out_vars = [dataset.variable_mapping[var] for var in cfg['out_vars']]
    y = y[:, idx_out_vars]
    #get target shape and land-sea fraction
    B, C, T, H, W = y.shape
    N = (B * C * T * ((H * W) - lsm.sum())).item()
    #get forecast
    prediction = model(x, horizon=horizon, context=context)
    #calculate loss
    crps = loss_fn(observation = y, prediction = prediction)
    mse = l2(prediction[:, 0], y)
    #set loss to 0 for land pixels
    crps *= (1 - lsm)[None, None, None, :, :]
    mse *= (1 - lsm)[None, None, None, :, :]
    #calculate discounted loss
    if mode == 'train':
        crps = crps.sum(dim = (0, 1, 3, 4)) * loss_decay[:T]
        mse = mse.sum(dim = (0, 1, 3, 4)) * loss_decay[:T]
    #calculate mean loss
    avg_crps = crps.sum().div(N)
    avg_rmse = mse.sum().div(N).sqrt()
    #training loss
    loss = avg_crps
    #log metrics
    logger = {}
    logger['rmse'] = avg_rmse.item()
    logger['crps'] = avg_crps.item()
    logger['loss'] = loss.item() 
    
    return loss, logger

# %%
# Main training loop 
# ======================================================================================
print(f"Training model {cfg['name']}", flush=True)

#Training helper variables
val_loss = 1e5
metric_tracker = []

for current_epoch in range(num_epochs):
    epoch_tracker = {}    
    val_loss_old = val_loss
    start_time = time.time()

    #Val loop
    model.eval()
    mode = 'val'
    metric_logger = {f"rmse": 0.0, f"crps":0.0, f"loss":0.0}
    with torch.no_grad():
        for i, batch in enumerate(val_dl):
            val_loss, logger = forward_step(batch, cfg, train_data, lsm, model, 
                                        loss_fn, loss_decay, l2, mode=mode)
            for key in metric_logger.keys():
                metric_logger[key] += logger[key]
        for key in metric_logger.keys():
            epoch_tracker[f"{mode}/{key}"] = metric_logger[key] / len(train_dl)

    #Val loop 2
    model.eval()
    mode = 'val2'
    metric_logger = {f"rmse": 0.0, f"crps":0.0, f"loss":0.0}
    with torch.no_grad():
        for i, batch in enumerate(val2_dl):
            loss, logger = forward_step(batch, cfg, train_data, lsm, model, 
                                        loss_fn, loss_decay, l2, mode=mode)
            for key in metric_logger.keys():
                metric_logger[key] += logger[key]
        for key in metric_logger.keys():
            epoch_tracker[f"{mode}/{key}"] = metric_logger[key] / len(train_dl)

    #Train loop
    model.train()
    mode = 'train'
    metric_logger = {f"rmse": 0.0, f"crps":0.0, f"loss":0.0}
    for i, batch in enumerate(train_dl):
        optim.zero_grad()
        with torch.cuda.amp.autocast(): 
            loss, logger = forward_step(batch, cfg, train_data, lsm, model, 
                                        loss_fn, loss_decay, l2, mode=mode)
        gradscaler.scale(loss).backward()
        gradscaler.step(optim)
        gradscaler.update()
        scheduler.step() #Warning can be ignored, is only caused by gradscaler.step() not being recognized as optimiser.step()
        for key in metric_logger.keys():
            metric_logger[key] += logger[key]
    for key in metric_logger.keys():
        epoch_tracker[f"{mode}/{key}"] = metric_logger[key] / len(train_dl)

    #Print progress
    print(f'Epoch: {current_epoch}, Train Loss: {epoch_tracker["train_loss"]:,.2e}, Val Loss: {epoch_tracker["val_loss"]:,.2e}', flush=True)
    print(f'Elapsed time:  {(time.time() - start_time)/60 :.2f} minutes', flush=True)

    # Wandb
    if cfg['use_wandb']:
        wandb.log(epoch_tracker)
    
    # Checkpoint
    torch.save(model.state_dict(), ckpt_path)
    if val_loss < val_loss_old:
        torch.save(model.state_dict(), best_ckpt_path)

    # Store metrics
    metric_tracker.append(epoch_tracker)
    buff = {key:[] for key in epoch_tracker.keys()}
    for d in metric_tracker:
        for key, val in d.items():
            buff[key].append(val)
    with open(os.path.join(model_dir, "metrics.json"), 'w') as file:
        json.dump(buff, file, cls=ExtendedJSONEncoder, indent=4)


# %%
