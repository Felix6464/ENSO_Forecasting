''' Distributed training of swinlstm basemodel on the CESM2 LENS dataset. 

@Author  :   Jakob Schlör 
@Time    :   2024/02/09 09:53:30
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import os, torch 
import xarray as xr
from torch.distributed.algorithms.join import Join

from s2aenso.distributed import trainer, config
from s2aenso.model import swinlstm, losses 
from s2aenso.utils import data, normalization

PATH = os.path.dirname(os.path.abspath(__file__))


class LENSTrainer(trainer.DistributedTrainer):

    def create_dataset(self):
        """
        Returns a tuple of (train_dl, val_dl).
        Will be available as self.train_dl and self.val_dl.
        These shall be iterators that yield batches.
        """
        # CERA-20C 
        normalizer = normalization.Normalizer_LENS(self.cfg.in_vars)
        self.train_data = data.load_reanalysis(root_dir = self.cfg.cera20c_dir,
                                               vars = self.cfg.in_vars, 
                                               sequence_length= self.cfg.sequence_length,
                                               normalizer=normalizer,
                                               split= 'train')
        
        val_data = data.load_cesm_lens(root_dir = self.cfg.lens_dir, 
                                       vars = self.cfg.in_vars, 
                                       sequence_length= self.cfg.sequence_length,
                                       split= 'val', 
                                       rank = self.rank,
                                       size = self.world_size,
                                       seed = self.cfg.seed)

        val2_data = data.load_reanalysis(root_dir = self.cfg.oras5_dir,
                                         vars = self.cfg.in_vars, 
                                         sequence_length= self.cfg.sequence_length,
                                         normalizer=normalizer,
                                         split= 'train')
        
        train_dl = torch.utils.data.DataLoader(self.train_data, 
                                            batch_size=self.cfg.batch_size, 
                                            shuffle=True, 
                                            drop_last=True,
                                            pin_memory=True,
                                            num_workers=self.cfg.num_workers)
        
        val_dl = torch.utils.data.DataLoader(val_data, 
                                          batch_size=self.cfg.batch_size, 
                                          shuffle=False, 
                                          drop_last=True,
                                          pin_memory=True,
                                          num_workers=self.cfg.num_workers)

        self.val2_dl = torch.utils.data.DataLoader(val2_data, 
                                                   batch_size=self.cfg.batch_size, 
                                                   shuffle=False, 
                                                   drop_last=True,
                                                   pin_memory=True,
                                                   num_workers=self.cfg.num_workers)

        # Add common land sea mask
        common_lsm = xr.open_dataset(self.cfg.lsm_path)['lsm'].data
        self.lsm = torch.as_tensor(common_lsm, device = self.device, dtype = torch.long)

        return train_dl, val_dl

    def create_model(self):
        """
        Returns a torch.nn.Module.
        Will be available as self.model.
        If you need multiple networks, e.g. for GANs, wrap them in a nn.Module.
        """
        model = swinlstm.SwinLSTMNet(
            input_dim = len(self.cfg.in_vars), 
            output_dim= len(self.cfg.out_vars),
            num_channels=self.cfg.num_channels,
            num_layers= self.cfg.num_layers,
            patch_size=self.cfg.patch_size,
            num_tails= self.cfg.num_tails,
            k_conv=self.cfg.k_conv,
            num_conditions= self.cfg.num_conditions
        )
        return model

    def create_loss(self):
        """
        Returns a loss function.
        Will be available as self.loss_fn.
        """
        decay = [self.cfg.decay_rate**i for i in range(self.cfg.sequence_length)] if self.cfg.decay_rate else [1 for _ in range(self.cfg.sequence_length)]
        self.loss_decay = torch.as_tensor(decay, device = self.device, dtype = torch.float)
        self.smooth_l1 = torch.nn.SmoothL1Loss(reduction = 'none')
        self.l2 = torch.nn.MSELoss(reduction = 'none')

        return losses.NormalCRPS(reduction="none", mode = self.cfg.loss_mode, dim = 1)

    def create_optimizer(self, params, lr):
        """
        Returns an optimizer.
        Will be available as self.optimizer.
        """
        optim = torch.optim.AdamW(params, lr = lr, weight_decay=self.cfg.weight_decay)
        return optim

    def create_scheduler(self, lr):
        """
        Returns a scheduler or None.
        """
        return torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                max_lr=lr, 
                                                steps_per_epoch=len(self.train_dl), 
                                                epochs=self.cfg.epochs, 
                                                pct_start=self.cfg.pct_start,
                                                last_epoch=-1,
                                                cycle_momentum=False)
        
    def evaluate_epoch(self):
        super().evaluate_epoch()
        #Join context manager prevents hangups due to uneven sharding
        with Join([self.model]):
            for batch_idx, batch in enumerate(self.val2_dl):
                with torch.no_grad():
                    _ = self.forward_step(batch_idx, batch, second_val=True)

    def forward_step(self, batch_idx, batch, **kwargs):
        """
        Performs a forward pass and returns the loss.
        """
        #unpack sample
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        timeseries, context = batch['x'], batch['month'] 
        # Randomly split the timeseries into history and horizon
        #history = torch.randint(1, cfg['max_history'], (1,))
        #horizon = cfg['sequence_length'] - history
        history = self.cfg.max_history 
        horizon = self.cfg.train_horizon if self.mode == 'train' else self.cfg.sequence_length - history 
        unused = self.cfg.sequence_length - history - horizon
        x, y, _ = timeseries.split([history, horizon, unused], dim = 2)

        # Select output variables
        idx_out_vars = [self.train_data.variable_mapping[var] for var in self.cfg.out_vars]
        y = y[:, idx_out_vars]
        #get target shape and land-sea fraction
        B, C, T, H, W = y.shape
        N = (B * C * T * ((H * W) - self.lsm.sum())).item()
        #get forecast
        prediction = self.model(x, horizon=horizon, context=context)
        #calculate loss
        crps = self.loss_fn(observation = y, prediction = prediction)
        mae = self.smooth_l1(prediction[:, 0], y)
        mse = self.l2(prediction[:, 0], y)
        #set loss to 0 for land pixels
        crps *= (1 - self.lsm)[None, None, None, :, :]
        mae *= (1 - self.lsm)[None, None, None, :, :]
        mse *= (1 - self.lsm)[None, None, None, :, :]
        #calculate discounted loss
        if self.mode == 'train':
            crps = crps.sum(dim = (0, 1, 3, 4)) * self.loss_decay[:T]
            mae = mae.sum(dim = (0, 1, 3, 4)) * self.loss_decay[:T]
            mse = mse.sum(dim = (0, 1, 3, 4)) * self.loss_decay[:T]
        #calculate mean loss
        avg_mae = mae.sum().div(N)    
        avg_crps = crps.sum().div(N)
        avg_rmse = mse.sum().div(N).sqrt()
        #training loss
        loss = avg_crps
        #log metrics
        if 'second_val' in kwargs.keys():
            self.current_metrics.log_metric('mae_val2', avg_mae.item())
            self.current_metrics.log_metric('rmse_val2', avg_rmse.item())
            self.current_metrics.log_metric('crps_val2', avg_crps.item())
            self.current_metrics.log_metric('loss_val2', loss.item())
        else:
            self.current_metrics.log_metric('mae', avg_mae.item())
            self.current_metrics.log_metric('rmse', avg_rmse.item())
            self.current_metrics.log_metric('crps', avg_crps.item())
            self.current_metrics.log_metric('loss', loss.item())
        
        return loss




def main():
    # Create a config
    slurm_id = os.environ.get('SLURM_JOB_ID', 0000000)
    params = {
        "name": f"base_swinlstm_cera20c_{slurm_id}",
        "cera20c_dir" : PATH + '/../../data/processed_data/cera-20c/',
        "lens_dir" : PATH + '/../../data/processed_data/cesm2_lens/historical/',
        "oras5_dir" : PATH + '/../../data/processed_data/oras5/',
        "lsm_path": PATH + "/../../data/processed_data/land_sea_mask_common.nc", 
        "model_dir" : PATH + "/../../models/basemodel_cera20c/",
        #data parameters
        "in_vars" : ['ssta', 'ssha'], #ssta ssha tauxa
        "out_vars" : ['ssta', 'ssha'],
        "sequence_length" : 28,
        "max_history" : 4,
        "train_horizon": 20,
        "num_workers" : 2,
        #training parameters
        "batch_size" : 16, 
        "base_batch_size": 8,
        "epochs" : 40,
        "base_lr" : 1e-4,
        "pct_start": 0.0,
        "base_beta1" : 0.9,
        "base_beta2" : 0.995,
        "weight_decay" : 1e-2,
        "decay_rate" : 0.65,
        "use_checkpointing" : True,
        "mixed_precision" : True,
        "scale_lr": True,
        "scale_beta1": False,
        "scale_beta2": False,
        "seed" : None,
        "clip_gradients" : 5.,
        "scheduler_step": "batch",
        "val_loss_name": "loss",
        "test_loss_name": "rmse",
        #model parameters
        "num_channels" : 256,
        "num_layers" : 2,
        "num_conditions" : 12,
        "num_tails" : 2,
        "patch_size" : (4, 4),
        "k_conv" : 7,
        "loss_mode" : "parametric",
        # logging parameters
        "use_wandb" : True,
        "wb_project" : 'S2A_Basemodels',
        "wb_tags": ['test'],
    }
    cfg = config.ExperimentConfig(params)
    # Create an experiment
    exp = LENSTrainer(cfg)
    # Run the experiment
    exp.train()

def test():
    # Create a config
    slurm_id = os.environ.get('SLURM_JOB_ID', 0000000)
    params = {
        "name": f"test_{slurm_id}",
        "cera20c_dir" : PATH + '/../../data/processed_data/cera-20c/',
        "lens_dir" : PATH + '/../../data/processed_data/cesm2_lens/historical/',
        "oras5_dir" : PATH + '/../../data/processed_data/oras5/',
        "lsm_path": PATH + "/../../data/processed_data/land_sea_mask_common.nc", 
        "model_dir" : PATH + "/../../models/basemodel_cera20c/",
        #data parameters
        "in_vars" : ['ssta', 'ssha'], #ssta ssha tauxa
        "out_vars" : ['ssta', 'ssha'],
        "sequence_length" : 28,
        "max_history" : 4,
        "train_horizon": 16,
        "num_workers" : 2,
        #training parameters
        "batch_size" : 2, 
        "base_batch_size": 8,
        "epochs" : 2,
        "base_lr" : 1e-3,
        "pct_start": 0.2,
        "base_beta1" : 0.9,
        "base_beta2" : 0.995,
        "weight_decay" : 1e-2,
        "decay_rate" : 0.65,
        "use_checkpointing" : True,
        "mixed_precision" : True,
        "scale_lr": True,
        "scale_beta1": False,
        "scale_beta2": False,
        "seed" : None,
        "clip_gradients" : 5.,
        "scheduler_step": "batch",
        "val_loss_name": "loss",
        "test_loss_name": "rmse",
        #model parameters
        "num_channels" : 256,
        "num_layers" : 2,
        "num_conditions" : 12,
        "num_tails" : 2,
        "patch_size" : (4, 4),
        "k_conv" : 7,
        "loss_mode" : "parametric",
        # logging parameters
        "use_wandb" : False,
        "wb_project" : 'S2A_Basemodels',
        "wb_tags": ['test'],
    }
    cfg = config.ExperimentConfig(params)
    # Create an experiment
    exp = LENSTrainer(cfg)
    # Run the experiment
    exp.train()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=ResourceWarning)
    main()
    #test()

