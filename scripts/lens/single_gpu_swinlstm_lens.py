# %%
import os, time, argparse, wandb, math
import xarray as xr
import torch 
from torch import nn
from importlib import reload
from matplotlib import pyplot as plt

from s2aenso.distributed.utils import *
from s2aenso.model import swinlstm, losses 
from s2aenso.utils import data, normalization, metric, utilities

PATH = os.path.dirname(os.path.abspath(__file__))
id = int(os.environ.get('SLURM_LOCALID', 0))
device= torch.device("cuda", id ) if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device: {device}', flush=True)
torch.cuda.empty_cache()

utilities.set_seed(42)

# Parameters
# ======================================================================================
slurm_id = os.environ.get('SLURM_JOB_ID', 0000000)
cfg = {
    "name": f"swinlstm_lens_{slurm_id}",
    "data_dir" : PATH + '/../../data/processed_data/cesm2_lens/historical_levels/temp_ocean/1_1_grid/final_ensembles/',
    "data_dir_picontrol" : PATH + '/../../data/processed_data/cesm2_lens/piControl/temp_ocean/processed/',
    "oras5_dir" : PATH + '/../../data/processed_data/enso_data_pacific/oras5/temp_ocean/1_1_grid/',
    "lsm_path": PATH + "/../../data/processed_data/enso_data_pacific/land_sea_mask_common.nc", 
    "model_dir" : PATH + "/../../models/swinlstm_lens/",
    "oras5_val": True,
    "decay" : True,
    "scheduler_paper": False,
    "one_cycle_lr": True,
    "probabilistic": True,
    "1_2_grid": True,
    "temp_ocean": True,
    "lower_levels": False,
    #training parameters
    "variable_loss_scaling": [1, 1, 1, 1, 1, 1, 1, 1, 1], #[1, 1, 1, 1, 1, 1, 1, 1, 1]
    "batch_size" : 8,           # Change for testing
    "epochs" : 40,              # Change for testing
    "base_lr" : 1e-4,
    "min_lr": 1e-6,
    "pct_start": (5000/220000),
    "div_factor": 1e10,
    "final_div_factor": 1,
    "d_size": 256,
    "warmup": 2000,
    "weight_decay" : 5e-2,
    "decay_rate" : 0.65,
    #model parameters
    "step_strided_conv": False,
    "cutout": 0.2,
    "sequence_length_train" : 32,
    "sequence_length_val" : 32,
    "input_length": 12,
    "output_length": 20,
    "num_channels" : 256,
    "num_layers" : 3,
    "num_conditions" : 12,
    "num_tails" : 2,
    "patch_size" : (4, 4),
    "k_conv" : 7,
    "loss_mode" : "parametric",
    #less important
    "device": device,
    "nino_loss": True,
    "gradscaler": True,
    "picontrol_val": False,
    "loss_mean_variant": False,
    #data parameters
    "in_vars" : ['temp_ocn_0a', 'temp_ocn_1a', 'temp_ocn_3a', 'temp_ocn_5a', 'temp_ocn_8a', 'temp_ocn_11a', 'temp_ocn_14a', 'tauxa', 'tauya'],
    "out_vars" : ['temp_ocn_0a', 'temp_ocn_1a', 'temp_ocn_3a', 'temp_ocn_5a', 'temp_ocn_8a', 'temp_ocn_11a', 'temp_ocn_14a', 'tauxa', 'tauya'],
    "num_workers" : 8,
    "lat_range": (5, 57),
    "lon_range": (18, 150),
    "lat_range_relative": (15, 35),
    "lon_range_relative": (40, 100),
    # logging parameters
    "use_wandb" : True,        # Change for testing
    "wb_project" : 'SwinLSTM_Paper',
    "wb_tags": ['single_gpu'],
    "rank": 0,
    "world_size": 1,
}

if cfg["1_2_grid"]:
    cfg["data_dir"] = PATH + '/../../data/processed_data/cesm2_lens/historical_levels/temp_ocean/1_2_grid/final_ensembles/'
    cfg["data_dir_picontrol"] = PATH + '/../../data/processed_data/cesm2_lens/piControl/temp_ocean_1_2_grid/processed/'
    cfg["oras5_dir"] = PATH + '/../../data/processed_data/enso_data_pacific/oras5/temp_ocean/1_2_grid/'
    cfg["lsm_path"] = PATH + "/../../data/processed_data/enso_data_pacific/land_sea_mask_common_1_2_grid.nc"
    cfg["lon_range"] = (0, 120)
    #cfg["lat_range_relative"] = (21, 31)
    #cfg["lon_range_relative"] = (46, 71)
    cfg["lat_range_relative"] = (15, 36)
    cfg["lon_range_relative"] = (49, 75)
    cfg["nino_3_lat"] = (21, 31)
    cfg["nino_3_lon"] = (56, 86) 
    cfg["nino_34_lon"] = (46, 71)
    cfg["nino_4_lon"] = (31, 56)
else:
    cfg["nino_3_lat"] = (21, 31)
    cfg["nino_3_lon"] = (62, 122) 
    cfg["nino_34_lon"] = (42, 92)
    cfg["nino_4_lon"] = (12, 62)

if cfg["temp_ocean"] == False:
    cfg["in_vars"] = ['ssta', 'ssha', 'tauxa']#, 'tauya']
    cfg["out_vars"] = ['ssta', 'ssha', 'tauxa']#, 'tauya']
    cfg["data_dir"] = PATH + '/../../data/processed_data/cesm2_lens/historical_levels/sst_ssh/final_ensembles/'
    cfg["oras5_dir"] = PATH + '/../../data/processed_data/enso_data_pacific/oras5/sst_ssh/'
    cfg["variable_loss_scaling"] = [1.0, 0.01, 0.01]

if cfg["lower_levels"]:
    cfg["data_dir"] = PATH + '/../../data/processed_data/cesm2_lens/historical_levels/temp_ocean/lower_levels/1_2_grid/final_ensembles/'
    cfg["data_dir_picontrol"] = PATH + '/../../data/processed_data/cesm2_lens/piControl/temp_ocean_1_2_grid/processed/lower_levels/'
    cfg["oras5_dir"] = PATH + '/../../data/processed_data/enso_data_pacific/oras5/temp_ocean/lower_levels/1_2_grid/'
    cfg["in_vars"] = ['temp_ocn_0a', 'temp_ocn_4a', 'temp_ocn_8a', 'temp_ocn_12a', 'temp_ocn_16a', 'temp_ocn_20a', 'temp_ocn_24a', 'tauxa', 'tauya'] #ssta ssha tauxa
    cfg["out_vars"] = ['temp_ocn_0a', 'temp_ocn_4a', 'temp_ocn_8a', 'temp_ocn_12a', 'temp_ocn_16a', 'temp_ocn_20a', 'temp_ocn_24a', 'tauxa', 'tauya'] #ssta ssha tauxa

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
parser.add_argument('-oras5', '--oras5_val', type=bool, help='Whether to use ORAS5 data for validation.')
parser.add_argument('-picontrol', '--picontrol_val', type=bool, help='Whether to use picontrol data for validation.')
parser.add_argument('-scheduler_paper', '--scheduler_paper', type=bool, help='Whether to us the scheduler from the paper.')
parser.add_argument('-1_2_grid', '--1_2_grid', type=bool, help='Whether to use ORAS5 data for validation.')
parser.add_argument('-probabilistic', '--probabilistic', type=bool, help='Whether to use ORAS5 data for validation.')
parser.add_argument('-decay', '--decay', type=bool, help='Whether to use ORAS5 data for validation.')
parser.add_argument('-epochs', '--epochs', type=int, help='Whether to use ORAS5 data for validation.')

config_parsed = vars(parser.parse_args())

cfg.update({k: v for k, v in config_parsed.items() if v is not None})

print("CONFIG", cfg)

# %%
# Load data
# ======================================================================================
reload(data)
reload(normalization)
normalizer = normalization.Normalizer_LENS(cfg['in_vars'], levels=cfg["temp_ocean"], low_levels=cfg["lower_levels"], grid=cfg["1_2_grid"])
normalizer_oras5 = None

train_data = data.load_cesm_lens(root_dir=cfg['data_dir'],
                                      vars=cfg['in_vars'],
                                      sequence_length=cfg['sequence_length_train'],
                                      split='train',
                                      rank=cfg['rank'],
                                      size=cfg['world_size'],
                                      num_directories_split=23,
                                      max_split=25,
                                      normalizer=normalizer)

print("LENGTH TRAIN DATA", len(train_data))

val_data = data.load_cesm_lens(root_dir=cfg['data_dir'],
                               vars=cfg['in_vars'],
                               sequence_length=cfg['sequence_length_val'],
                               split='val',
                               rank=cfg['rank'],
                               size=cfg['world_size'],
                               num_directories_split=23,
                               max_split=25,
                               normalizer=normalizer)

print("LENGTH VAL DATA", len(val_data))

train_dl = torch.utils.data.DataLoader(train_data,
                                       batch_size=cfg['batch_size'],
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=True,
                                       num_workers=cfg['num_workers'])

print("LENGTH TRAIN DL", len(train_dl))

val_dl = torch.utils.data.DataLoader(val_data,
                                     batch_size=cfg['batch_size'],
                                     shuffle=False,
                                     drop_last=True,
                                     pin_memory=True,
                                     num_workers=cfg['num_workers'])

if cfg["picontrol_val"]:
    val_data_picontrol = data.load_picontrol(root_dir = cfg['data_dir_picontrol'],
                                    vars = cfg['in_vars'], 
                                    sequence_length= cfg['sequence_length_train'],
                                    normalizer=normalizer,
                                    split= 'val',
                                    test=False)

    val_dl_picontrol = torch.utils.data.DataLoader(val_data_picontrol,
                                        batch_size=cfg['batch_size'],
                                        shuffle=False,
                                        drop_last=True,
                                        pin_memory=True,
                                        num_workers=cfg['num_workers'])

if cfg['oras5_val']:
    # ORAS5 
    val2_data = data.load_reanalysis(root_dir = cfg['oras5_dir'],
                                    vars = cfg['in_vars'], 
                                    sequence_length = cfg['sequence_length_val'],
                                    normalizer=normalizer_oras5,
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
    num_conditions= cfg['num_conditions'],
    cutout=cfg["cutout"],
    step_strided_conv=cfg["step_strided_conv"]
).to(device)

print("Number of Parameters of Model: ", utilities.count_parameters(model))

# %%
# Loss
# ======================================================================================
decay = [cfg['decay_rate']**i for i in range(cfg['sequence_length_train'])] if cfg['decay_rate'] else [1 for _ in range(cfg['sequence_length_train'])]
loss_decay = torch.as_tensor(decay, device = device, dtype = torch.float)
smooth_l1 = torch.nn.SmoothL1Loss(reduction = 'none')
l2 = torch.nn.MSELoss(reduction = 'none')

loss_fn =  losses.NormalCRPS(reduction="none", mode = cfg['loss_mode'], dim = 1)

# %%
# Optimizer and Scheduler
# ======================================================================================
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

# %%
# Create model directory
# ======================================================================================
model_dir = os.path.join(cfg['model_dir'], cfg['name'])
ckpt_path = os.path.join(model_dir, 'cpkt.pth') 
best_ckpt_path = os.path.join(model_dir, 'best.pth') 
best_ckpt_path_oras5 = os.path.join(model_dir, 'best_oras5.pth')
best_ckpt_path_picontrol = os.path.join(model_dir, 'best_picontrol.pth')


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
# %%
def forward_step(batch, dataset, cfg: dict, lsm, model, loss_fn, loss_decay, l2, 
                 mode: str='train'):
    """
    Performs a forward pass and returns the loss.
    """
    #unpack sample
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    timeseries, context = batch['x'], batch['month'] 
    
    # Randomly split the timeseries into history and horizon
    #history = torch.randint(1, cfg['max_history'], (1,))
    history = cfg['input_length']
    horizon = cfg['output_length']


    predictor, predictand = timeseries.split([history, horizon], dim = 2)
    # Predictor shape: [8, 7, 12, 64, 160]
    # Predictand shape: [8, 7, 20, 64, 160]

    # Select lat lon range
    predictor = predictor[:, :, :,
                    cfg['lat_range'][0]:cfg['lat_range'][1],
                    cfg['lon_range'][0]:cfg['lon_range'][1]]
    
    predictand = predictand[:, :, :,
                    cfg['lat_range'][0]:cfg['lat_range'][1],
                    cfg['lon_range'][0]:cfg['lon_range'][1]]
    
    lsm = lsm[cfg['lat_range'][0]:cfg['lat_range'][1], cfg['lon_range'][0]:cfg['lon_range'][1]]


    # Select output variables
    #idx_out_vars = [dataset.datasets[0].variable_mapping[var] for var in cfg['out_vars']]
    #predictand = predictand[:, idx_out_vars]

    #get target shape and land-sea fraction
    B, C, T, H, W = predictand.shape
    N = (B * C * T * ((H * W) - lsm.sum())).item()

    #get forecast
    print("Predictor shape: ", predictor.shape)
    print("Predictand shape: ", predictand.shape)
    print("Context: ", context)
    print("Context shape: ", context.shape)
    prediction = model(predictor, horizon=horizon, context=context)
    # Prediction shape: [32, 2, 7, 20, 64, 160]
    # y shape: [32, 7, 20, 64, 160]

    # Calculate Nino Losss
    if cfg["nino_loss"]:
        nino_pred = prediction[:, :, 0, :, :, :]
        nino_pred *= (1 - lsm)[None, None, None, :, :]
        nino_pred = nino_pred[
                                :,
                                :,
                                :,
                                cfg["lat_range_relative"][0] : cfg["lat_range_relative"][1],
                                cfg["lon_range_relative"][0] : cfg["lon_range_relative"][1],
                                ]
        nino_pred = nino_pred.mean(dim = (3, 4))    

        nino_true = predictand[:, 0, :, :, :]
        nino_true *= (1 - lsm)[None, None, :, :]
        nino_true = nino_true[
                                :,
                                :,
                                cfg["lat_range_relative"][0] : cfg["lat_range_relative"][1],
                                cfg["lon_range_relative"][0] : cfg["lon_range_relative"][1],
                            ]
        nino_true = nino_true.mean(dim = (2, 3))

        loss_nino = losses.loss_nino(nino_pred[:, 0], nino_true.float())
        nino_score = losses.cal_ninoscore(cfg["ninoweight"], nino_pred[:, 0], nino_true.float())

    if mode == 'val':
        prediction = prediction[:, :, 0, :, :, :]
        predictand = predictand[:, 0, :, :, :]
        N = N / len(cfg["in_vars"])

        # Calculate Nino Index
        nino_pred_34 = prediction[:, :, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
        nino_true_34 = predictand[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]

        nino_pred_4 = prediction[:, :, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]
        nino_true_4 = predictand[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]

        # calculate ACC
        acc_mean = metric.anomaly_correlation_coefficient([(prediction[:, 0], None)], [(predictand, None)], None, None, None)
        acc_mean_34 = metric.anomaly_correlation_coefficient([(nino_pred_34[:, 0], None)], [(nino_true_34, None)], None, None, None)
        acc_mean_4 = metric.anomaly_correlation_coefficient([(nino_pred_4[:, 0], None)], [(nino_true_4, None)], None, None, None)

        crps_34 = loss_fn(observation = nino_true_34, prediction = nino_pred_34)
        crps_4 = loss_fn(observation = nino_true_4, prediction = nino_pred_4)
        mse_34 = l2(nino_pred_34[:, 0], nino_true_34)
        mse_4 = l2(nino_pred_4[:, 0], nino_true_4)
    
    #calculate loss
    crps = loss_fn(observation = predictand, prediction = prediction)
    mse = l2(prediction[:, 0], predictand)

    if cfg["1_2_grid"]:

        if len(predictand.shape) == 4:
            pred_sliced = prediction[:, :, :, :, 29:95]
            targ_sliced = predictand[:, :, :, 29:95]
        else:
            pred_sliced = prediction[:, :, :, :, :, 29:95]
            targ_sliced = predictand[:, :, :, :, 29:95]
        
        lsm_sliced = lsm[:, 29:95]

        crps_sliced = loss_fn(observation = targ_sliced, prediction = pred_sliced)
        mse_sliced = l2(pred_sliced[:, 0], targ_sliced)

        if len(crps_sliced.shape) == 4:
            crps_sliced = crps_sliced.unsqueeze(0)
            mse_sliced = mse_sliced.unsqueeze(0)

        crps_sliced *= (1 - lsm_sliced)[None, None, None, :, :]
        mse_sliced *= (1 - lsm_sliced)[None, None, None, :, :]

        avg_crps_sliced = crps_sliced.sum().div(N)
        avg_rmse_sliced = mse_sliced.sum().div(N).sqrt()

    if cfg["nino_loss"]:
        mse_nino = l2(nino_pred[:, 0], nino_true)
        crps_nino = loss_fn(observation = nino_true, prediction = nino_pred)
    # CRPS shape val: [32, 24, 64, 160]
    # MSE shape val: [32, 24, 64, 160]

    if len(crps.shape) == 4:
        crps = crps.unsqueeze(0)
        mse = mse.unsqueeze(0)
        #maybe an issue with the loss function

    # Apply land-sea mask
    #set loss to 0 for land pixels
    crps *= (1 - lsm)[None, None, None, :, :]
    mse *= (1 - lsm)[None, None, None, :, :]

    if mode == "train":
        # Define weights for each variable in the loss
        #CRPS shape torch.Size([8, 9, 20, 52, 120])
        #MSE shape torch.Size([8, 9, 20, 52, 120])
        #Variable weights shape torch.Size([1, 9, 1, 1, 1])
        variable_weights = torch.tensor(cfg["variable_loss_scaling"], device=device).view(1, -1, 1, 1, 1)
        
        # Apply weights to the loss
        crps *= variable_weights
        mse *= variable_weights

    if cfg["decay"]:
        #calculate discounted loss
        if mode == 'train':
            # CRPS shape: [32, 7, 20, 64, 160]
            # MSE shape: [32, 7, 20, 64, 160]
            # DECAY shape: [20]
            crps = crps.sum(dim = (0, 1, 3, 4)) * loss_decay[:T]
            mse = mse.sum(dim = (0, 1, 3, 4)) * loss_decay[:T]

            if cfg["nino_loss"]:
                mse_nino = mse_nino.sum(dim = (0, 1)) * loss_decay[:T]

    #calculate mean loss
    if cfg["loss_mean_variant"]:
        avg_crps = crps.mean()
        avg_rmse = mse.mean().sqrt()
        avg_mse = mse.mean()
        if cfg["nino_loss"]:
            avg_rmse_nino = mse_nino.mean().sqrt()
            avg_crps_nino = crps_nino.mean()
    else:
        avg_crps = crps.sum().div(N)
        avg_rmse = mse.sum().div(N).sqrt()
        avg_mse = mse.sum().div(N)
        if cfg["nino_loss"]:
            #avg_rmse_nino = mse_nino.sum().div(N).sqrt()
            #avg_crps_nino = crps_nino.sum().div(N)
            avg_crps_nino = crps_nino.mean()
            avg_rmse_nino = mse_nino.mean().sqrt()



        if mode == "val":
            avg_crps_34 = crps_34.sum().div(N)
            avg_crps_4 = crps_4.sum().div(N)
            avg_rmse_34 = mse_34.sum().div(N).sqrt()
            avg_rmse_4 = mse_4.sum().div(N).sqrt()

    
    if cfg["probabilistic"]:
        loss = avg_crps
        #training loss
        if cfg["nino_loss"]:
            #loss_nino = losses.log_scale_loss(loss_nino)
            loss_nino = avg_rmse_nino / 10
            #loss_nino = avg_rmse_nino
            loss = loss + loss_nino.to(cfg["device"])
    else:
        loss = avg_rmse

        if cfg["nino_loss"]:
            #loss_nino = losses.log_scale_loss(loss_nino)
            loss_nino = avg_rmse_nino / 10
            loss = loss + loss_nino.to(cfg["device"])

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

    if cfg["1_2_grid"]:
        logger["rmse_avg_sliced"] = avg_rmse_sliced.item()
        logger["crps_avg_sliced"] = avg_crps_sliced.item()
    
    return loss, logger

# %%
# Main training loop 
# ======================================================================================
print(f"Training model {cfg['name']}", flush=True)

#Training helper variables
val_loss = 1e5
val_loss_oras5 = 1e5
val_loss_picontrol = 1e5
best_val_loss = 1e4
best_val_loss_oras5 = 1e4
best_val_loss_picontrol = 1e4

metric_tracker = []
flag_print = False


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
    
    if cfg["1_2_grid"]:
        metric_logger["rmse_avg_sliced"] = 0.0
        metric_logger["crps_avg_sliced"] = 0.0


    with torch.no_grad():
        for i, batch in enumerate(val_dl):
            val_loss, logger = forward_step(batch, train_data, cfg, lsm, model, 
                                            loss_fn, loss_decay, l2, mode=mode)
            #print("Loss val: ", loss.item())
            for key in metric_logger.keys():
                    metric_logger[key] += logger[key]

        for key in metric_logger.keys():
            epoch_tracker[f"{mode}/{key}"] = metric_logger[key] / len(val_dl)
            print(f"{mode}/{key}: ", epoch_tracker[f"{mode}/{key}"])

    if cfg["picontrol_val"]:
        print("Starting Picontrol-Val Loop")
        model.eval()
        mode = 'val'
        if cfg["nino_loss"]:
            metric_logger = {f"rmse_avg": 0.0, f"rmse_avg_34": 0.0, f"rmse_avg_4": 0.0, f"crps_avg": 0.0, f"crps_avg_34": 0.0, f"crps_avg_4": 0.0, f"loss_avg":0.0, f"mse_avg": 0.0, f"nino_score": 0.0, f"nino_loss": 0.0, f"acc_mean": 0.0}
        else:
            metric_logger = {f"rmse_avg": 0.0, f"rmse_avg_34": 0.0, f"rmse_avg_4": 0.0, f"crps_avg": 0.0, f"crps_avg_34": 0.0, f"crps_avg_4": 0.0, f"loss_avg":0.0, f"mse_avg": 0.0, f"acc_mean": 0.0}
        
        if cfg["1_2_grid"]:
            metric_logger["rmse_avg_sliced"] = 0.0
            metric_logger["crps_avg_sliced"] = 0.0

        with torch.no_grad():
            for i, batch in enumerate(val_dl_picontrol):
                val_loss_picontrol, logger = forward_step(batch, train_data, cfg, lsm, model, 
                                                loss_fn, loss_decay, l2, mode=mode)
                #print("Loss val: ", loss.item())
                for key in metric_logger.keys():
                        metric_logger[key] += logger[key]

            for key in metric_logger.keys():
                epoch_tracker[f"{mode}/{key}_val_picontrol"] = metric_logger[key] / len(val_dl_picontrol)
                print(f"{mode}/{key}: ", epoch_tracker[f"{mode}/{key}"])

    if cfg['oras5_val']:
        #Val loop 2
        print("Starting Val2 Loop")
        model.eval()
        mode = 'val'
        if cfg["nino_loss"]:
            metric_logger = {f"rmse_avg": 0.0, f"rmse_avg_34": 0.0, f"rmse_avg_4": 0.0, f"crps_avg": 0.0, f"crps_avg_34": 0.0, f"crps_avg_4": 0.0, f"loss_avg":0.0, f"mse_avg": 0.0, f"nino_score": 0.0, f"nino_loss": 0.0, f"acc_mean": 0.0}
        else:
            metric_logger = {f"rmse_avg": 0.0, f"rmse_avg_34": 0.0, f"rmse_avg_4": 0.0, f"crps_avg": 0.0, f"crps_avg_34": 0.0, f"crps_avg_4": 0.0, f"loss_avg":0.0, f"mse_avg": 0.0, f"acc_mean": 0.0}
        
        if cfg["1_2_grid"]:
            metric_logger["rmse_avg_sliced"] = 0.0
            metric_logger["crps_avg_sliced"] = 0.0

        with torch.no_grad():
        #    print("Length VAL2 dl: ", len(val2_dl))
            for i, batch in enumerate(val2_dl):

                val_loss_oras5, logger = forward_step(batch, train_data, cfg, lsm, model, 
                                            loss_fn, loss_decay, l2, mode=mode)
                #print("Loss val2: ", loss.item())
                for key in metric_logger.keys():
                    metric_logger[key] += logger[key]


        for key in metric_logger.keys():
            epoch_tracker[f"{mode}/{key}_val_oras5"] = metric_logger[key] / len(val2_dl)
            print(f"{mode}/{key}: ", epoch_tracker[f"{mode}/{key}"])


    #Train loop
    print("Starting Train Loop")
    model.train()
    mode = 'train'
    if cfg["nino_loss"]:
        metric_logger = {f"rmse_avg": 0.0, f"crps_avg":0.0, f"loss_avg":0.0, f"mse_avg": 0.0, f"nino_score": 0.0, f"nino_loss": 0.0}
    else:
        metric_logger = {f"rmse_avg": 0.0, f"crps_avg":0.0, f"loss_avg":0.0, f"mse_avg": 0.0}
    
    if cfg["1_2_grid"]:
        metric_logger["rmse_avg_sliced"] = 0.0
        metric_logger["crps_avg_sliced"] = 0.0

    for i, batch in enumerate(train_dl):

        if cfg["scheduler_paper"]: 
            optim.optimizer.zero_grad()
        else: 
            optim.zero_grad()


        with torch.cuda.amp.autocast(enabled=cfg["gradscaler"]):

            loss, logger = forward_step(batch, train_data, cfg, lsm, model, 
                        loss_fn, loss_decay, l2, mode=mode)
            
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
    print(f'Epoch: {current_epoch}, Train Loss: {epoch_tracker["train/loss_avg"]:,.2e}, '
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
    
    if val_loss_oras5 < best_val_loss_oras5:
        best_val_loss_oras5 = val_loss_oras5
        print("Saving new best model oras5")
        torch.save(model.state_dict(), best_ckpt_path_oras5)
    
    if val_loss_picontrol < best_val_loss_picontrol:
        best_val_loss_picontrol = val_loss_picontrol
        print("Saving new best model picontrol")
        torch.save(model.state_dict(), best_ckpt_path_picontrol)

    # Store metrics
    metric_tracker.append(epoch_tracker)
    buff = {key:[] for key in epoch_tracker.keys()}
    for d in metric_tracker:
        for key, val in d.items():
            buff[key].append(val)
    with open(os.path.join(model_dir, "metrics.json"), 'w') as file:
        json.dump(buff, file, cls=ExtendedJSONEncoder, indent=4)