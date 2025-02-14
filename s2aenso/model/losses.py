''' Collection of loss functions for training the model. 

@Author  :   Jakob Schlör, Jannik Thuemmel 
@Time    :   2023/08/16 15:28:05
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import numpy as np
import torch
import torch.nn as nn

class GammaWeighting(nn.Module):    
    
    def __init__(self, gamma_start, gamma_end, rampup_epochs, device):
        """ Gamma Weighting of loss.

        Implementation by @sebastianhofmann
        
        Usage:
        >>> gamma = gamma_scheduler(loss.shape[0], epoch)
        >>> gamma /= gamma.sum()
        >>> loss = (loss * gamma).sum()

        Args:
            gamma_start (_type_): _description_
            gamma_end (_type_): _description_
            rampup_epochs (_type_): _description_
            device (_type_): _description_
        """
        super().__init__()
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.rampup_epochs = rampup_epochs
        self.device = device
    
    def forward(self, steps, epoch):
        gamma = (
            self.gamma_start 
            + (self.gamma_end - self.gamma_start) * min(epoch / self.rampup_epochs, 1)
        )
        return torch.pow(gamma, torch.arange(steps, device=self.device))


def get_statistics(prediction: torch.Tensor, dim: int = 1, mode: str = 'ensemble', epsilon: float = 1e-9):
    '''Compute the mean and standard deviation of the predictive distribution

    Author: @jannikthuemmel
    Args:
         prediction: (batch, ensemble, *) tensor of ensemble predictions
         dim: the dimension of the ensemble
         mode: the type of predictive distribution, can be 'ensemble', 'parametric' or 'sample'
         epsilon: a small number to add to the standard deviation to avoid numerical instability
    Returns:
        mu, sigma     (batch, *) tensors of mean and standard deviation
    '''
    #print("prediction: ", prediction.shape)
    if mode == 'ensemble':
        mu, sigma = prediction.mean(dim = dim), prediction.std(dim = dim) #mean and standard deviation of the ensemble
    elif mode == 'parametric':
        mu, sigma = prediction.split(1, dim = dim)
        mu, sigma = mu.squeeze(dim=dim), sigma.squeeze(dim=dim)
    elif mode == 'sample':
        mu, sigma = prediction, torch.ones_like(prediction)
    else:
        raise NotImplementedError(f'Mode {mode} not implemented')

    return mu, sigma + epsilon


class NormalCRPS(nn.Module):
    '''Continuous Ranked Probability Score (CRPS) loss for a normal distribution
    as described in the paper "Probabilistic Forecasting with Gated Neural Networks".
    
    Implementation by @jannikthuemmel
    '''
    def __init__(self, reduction: str = 'mean', dim: int = 1,  
                 mode: str = 'ensemble'):
        '''
        reduction: the reduction method to use, can be 'mean', 'sum' or 'none'
        sigma_transform: the transform to apply to the std estimate, can be 'softplus', 'exp' or 'none'
        '''
        super().__init__()
        self.dim, self.mode = dim, mode

        self.sqrtPi = torch.as_tensor(np.pi).sqrt()
        self.sqrtTwo = torch.as_tensor(2.).sqrt()

        if reduction == 'mean':
            self.reduce = lambda x: x.mean()
        elif reduction == 'sum':
            self.reduce = lambda x: x.sum()
        elif reduction == 'none':
            self.reduce = lambda x: x
        else:
            raise NotImplementedError(f'Reduction {reduction} not implemented')

    def forward(self, observation: torch.Tensor, prediction: torch.Tensor):
        '''
        Compute the CRPS for a normal distribution
            :param observation: (batch, *) tensor of observations
            :param mu: (batch, *) tensor of means
            :param log_sigma: (batch, *) tensor of log standard deviations
            :return: CRPS score     
            '''
        mu, sigma = get_statistics(prediction, mode=self.mode, dim=self.dim)
        #print("mu: ", mu.shape)
        #print("sigma: ", sigma.shape)
        #print("observation: ", observation.shape)

        z = (observation - mu) / sigma #z transform
        phi = torch.exp(-z ** 2 / 2).div(self.sqrtTwo * self.sqrtPi) #standard normal pdf
        score = sigma * (z * torch.erf(z / self.sqrtTwo) + 2 * phi - 1 / self.sqrtPi) #crps as per Gneiting et al 2005
        reduced_score = self.reduce(score)
        return reduced_score
    

class EmpiricalCRPS(nn.Module):
    '''Continuous Ranked Probability Score (CRPS) loss for empirical distribution.

    Gneiting, Raftery (2012), https://doi.org/10.1198/016214506000001437

    Args:
        reduction (str, optional): Reduction over samples in batch. Defaults to 'mean'.
    '''
    def __init__(self, dim: int = 1,  reduction = 'mean'):
        super().__init__()

        self.dim = dim
        if reduction == 'mean':
            self.reduce = lambda x: x.nanmean()
        elif reduction == 'sum':
            self.reduce = lambda x: x.nansum()
        elif reduction == 'none':
            self.reduce = lambda x: x
        else:
            raise NotImplementedError(f'Reduction {reduction} not implemented')
    
    def forward(self, observation: torch.Tensor, prediction: torch.Tensor):
        """CPRS for empirical distribution.

        Args:
            observation (torch.Tensor): Target data of shape (batch, n_member, *)
            prediction (torch.Tensor): Predicted data of shape (batch, n_observation, *)

        Returns:
            cprs_nrg (torch.Tensor): CRPS for each feature (batch, *)
        """
        n_member = prediction.shape[self.dim]
        absolute_error = torch.mean(torch.abs(prediction - observation), dim=self.dim)

        prediction_sort = torch.sort(prediction, dim=self.dim).values
        diff = torch.diff(prediction_sort, dim=self.dim)
        weight = torch.arange(
                1, n_member, dtype=torch.float32, device=prediction.device
            ) * torch.arange(
                n_member - 1, 0, -1, dtype=torch.float32, device=prediction.device
            )
        # Expand dimensions of weight
        ndim = [1] * prediction_sort.ndim
        ndim[self.dim] = len(weight) 
        weight = weight.reshape(ndim)

        score = absolute_error - torch.sum(diff * weight, dim=self.dim) / n_member**2
        reduced_score = self.reduce(score)
        return reduced_score
    

class lrwarm:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
    

def cal_ninoscore(ninoweight, y_pred, y_true):
    # compute Nino score
    with torch.no_grad():
        pred = y_pred - y_pred.mean(dim=0, keepdim=True)
        true = y_true - y_true.mean(dim=0, keepdim=True)
        cor = (pred * true).sum(dim=0) / (
            torch.sqrt(torch.sum(pred ** 2, dim=0) * torch.sum(true ** 2, dim=0))
            + 1e-6
        )
        acc = (ninoweight * cor).sum()
        rmse = torch.mean((y_pred - y_true) ** 2, dim=0).sqrt().sum()
        sc = 2 / 3.0 * acc - rmse
    return sc.item()

def loss_var(y_pred, y_true):
    if len(y_pred.shape) == 5:
        rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])
    else:
        rmse = torch.mean((y_pred - y_true) ** 2, dim=[2, 3])
    rmse = rmse.sqrt().mean(dim=0)
    if len(rmse.shape) == 2:
        rmse = torch.sum(rmse, dim=[0, 1])
    else:
        rmse = torch.sum(rmse)
    return rmse

def loss_nino(y_pred, y_true):
    # with torch.no_grad():
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
    return rmse.sum()

def loss_nino_crps(y_pred, y_true):
    # with torch.no_grad():
    crps = torch.mean((y_pred - y_true) ** 2, dim=0)
    return crps.sum()

def log_scale_loss(loss):
    """
    Apply log scaling to a tensor loss value. Preserves the sign of the original loss.
    Adjusted for tensors on CUDA device.

    Parameters:
    - loss (torch.Tensor): The original loss tensor to scale.

    Returns:
    - torch.Tensor: The log-scaled loss tensor.
    """
    # Ensure the tensor is on CPU if it's on CUDA
    if loss.is_cuda:
        loss = loss.cpu()

    # Use torch operations instead of numpy to avoid unnecessary conversions
    scaled_loss = torch.sign(loss) * torch.log1p(torch.abs(loss))
    return scaled_loss