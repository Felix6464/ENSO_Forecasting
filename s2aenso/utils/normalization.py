'''Functions and classes to create dataloading of CESM2 LENS data. 

@Author  :   Sebastian Hofmann, Jannik Thuemmel, Jakob Schlör 
@Time    :   2024/02/05 15:00:44
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import torch
import xarray as xr


# Normalization for CESM-LENS
# =============================================================================

# Normalization for CESM-LENS over all 100 ensembles
STATS_ = {
    'ssha': {'mean': -0.07083344631246291, 'std': 4.4444851147883755},
    'ssta': {'mean': 0.04846862995265318, 'std': 0.7035480662957583},
    'tauxa': {'mean': 0.0006506717658453454, 'std': 0.2024029003837687},
}

# Normalization for CESM-LENS over 25 ensembles
STATS = {
    'ssha': {'mean': -0.039290446946714706e-17, 'std': 4.8042421750056405},
    'ssta': {'mean': 0.008735961004897047e-17, 'std': 0.7581828924193575},
    'tauxa': {'mean': -8.286767092028676e-20, 'std': 0.21429222974602877}, #-0.0008286767092028676
    'tauya': {'mean': -6.907210705671194e-20, 'std': 0.1329186389206408},
}


# Normalization for CESM-LENS 1_2_grid resolution over 25 ensembles
STATS_LEVELS_GRID_1_2 = {
    'temp_ocn_0a': {'mean': -1.460749187956891e-17, 'std': 0.6716897243585809},
    'temp_ocn_1a': {'mean': 6.588296232102237e-18, 'std': 0.6718217430638053},
    'temp_ocn_3a': {'mean': 6.0490856156816606e-18, 'std': 0.7230672257454342},
    'temp_ocn_5a': {'mean': -1.4047735896765925e-17, 'std': 0.8418841484002196},
    'temp_ocn_8a': {'mean': -1.89722131875779e-18, 'std': 0.9127529247324552},
    'temp_ocn_11a': {'mean': 4.809813862225761e-19, 'std': 0.9509003051051766},
    'temp_ocn_14a': {'mean': -1.1246988115251004e-17, 'std': 0.9060856579963922},
    'tauxa': {'mean': -3.9094611618755137e-20, 'std': 0.19904808837784083},
    'tauya': {'mean': 1.7936438524139273e-20, 'std': 0.1323912433010137},
}

# Normalization for CESM-LENS 1_2_grid resolution over 100 ensembles
STATS_LEVELS_GRID_1_2_ALL = {
    'temp_ocn_0a': {'mean': -1.460749187956891e-17, 'std': 0.6707193411666164},
    'temp_ocn_1a': {'mean': 6.588296232102237e-18, 'std': 0.6708457100967729},
    'temp_ocn_3a': {'mean': 6.0490856156816606e-18, 'std': 0.7219668185682295},
    'temp_ocn_5a': {'mean': -1.4047735896765925e-17, 'std': 0.8409158748930468},
    'temp_ocn_8a': {'mean': -1.89722131875779e-18, 'std': 0.911882499347164},
    'temp_ocn_11a': {'mean': 4.809813862225761e-19, 'std': 0.9499027942590248},
    'temp_ocn_14a': {'mean': -1.1246988115251004e-17, 'std': 0.9049123856449006},
    'tauxa': {'mean': -3.9094611618755137e-20, 'std': 0.19905465166576986},
    'tauya': {'mean': 1.7936438524139273e-20, 'std': 0.13234788275774506},
}

# Normalization for CESM-LENS over 25 ensembles
STATS_LEVELS = {
    'temp_ocn_0a': {'mean': -1.460749187956891e-17, 'std': 0.6716897243585809},
    'temp_ocn_1a': {'mean': 6.588296232102237e-18, 'std': 0.6718217430638053},
    'temp_ocn_3a': {'mean': 6.0490856156816606e-18, 'std': 0.7230672257454342},
    'temp_ocn_5a': {'mean': -1.4047735896765925e-17, 'std': 0.8418841484002196},
    'temp_ocn_8a': {'mean': -1.89722131875779e-18, 'std': 0.9127529247324552},
    'temp_ocn_11a': {'mean': 4.809813862225761e-19, 'std': 0.9509003051051766},
    'temp_ocn_14a': {'mean': -1.1246988115251004e-17, 'std': 0.9060856579963922},
    'tauxa': {'mean': -3.9094611618755137e-20, 'std': 0.19904808837784083},
    'tauya': {'mean': 1.7936438524139273e-20, 'std': 0.1323912433010137},
}

# Normalization for CESM-LENS over 25 ensembles
STATS_LOW_LEVELS = {
    'temp_ocn_0a': {'mean': -6.698876866794135e-18, 'std': 0.665163241444853},
    'temp_ocn_4a': {'mean': -1.0407765992162461e-18, 'std': 0.7822344517903749},
    'temp_ocn_8a': {'mean': 8.541395261731543e-18, 'std': 0.9079767637025667},
    'temp_ocn_12a': {'mean': -1.4047735896765925e-17, 'std': 0.8409158748930469},
    'temp_ocn_16a': {'mean': -1.9769443837913937e-18, 'std': 0.8060836785913406},
    'temp_ocn_20a': {'mean': -1.9949220876653464e-18, 'std': 0.5367531272750813},
    'temp_ocn_24a': {'mean': 7.2953425503073835e-19, 'std': 0.336298634709828},
    'tauxa': {'mean': -3.9094611618755137e-20, 'std': 0.19905465166576986},
    'tauya': {'mean': 1.7936438524139273e-20, 'std': 0.13234788275774506},
}

def build_means_levels(variables: list, low_levels=False, grid=False):
    means = torch.zeros(len(variables))
    for i, var in enumerate(variables):
        if not low_levels:
            if grid == True:
                means[i] = STATS_LEVELS_GRID_1_2[var]['mean']
            else:
                means[i] = STATS_LEVELS[var]['mean']
        else:
            means[i] = STATS_LOW_LEVELS[var]['mean']
    return means


def build_stds_levels(variables: list, low_levels=False, grid=False):
    stds = torch.zeros(len(variables))
    for i, var in enumerate(variables):
        if not low_levels:
            if grid == True:
                stds[i] = STATS_LEVELS_GRID_1_2[var]['std']
            else:
                stds[i] = STATS_LEVELS[var]['std']
        else:
            stds[i] = STATS_LOW_LEVELS[var]['std']
    print("Variable: ", var, " Mean: ", STATS_LEVELS[var]['mean'], " Std: ", STATS_LEVELS[var]['std'])
    return stds


def build_means(variables: list):
    means = torch.zeros(len(variables))
    for i, var in enumerate(variables):
        means[i] = STATS[var]['mean']
    return means


def build_stds(variables: list):
    stds = torch.zeros(len(variables))
    for i, var in enumerate(variables):
        stds[i] = STATS[var]['std']
        print("Variable: ", var, " Mean: ", STATS[var]['mean'], " Std: ", STATS[var]['std'])
    return stds


def normalize_tensor(x, means, stds):
    assert x.dim() >= 2
    indices = (None,) + (slice(None),) + (None,) * (x.dim() - 2)
    if not torch.allclose(means[indices], torch.zeros_like(means[indices]), atol=1e-7):
        return (x - means[indices]) / stds[indices]
    else:
        print('Warning: mean is zero')
        return x / stds[indices]


def denormalize_tensor(x, means, stds, device=None):
    assert x.dim() >= 2
    indices = (None,) + (slice(None),) + (None,) * (x.dim() - 2)
    if not torch.allclose(means[indices], torch.zeros_like(means[indices]), atol=1e-7):
        if device is not None:
            return x.to(device) * stds[indices].to(device) + means[indices].to(device)
        else:
            return x * stds[indices] + means[indices]
    else:
        if device is not None:
            return x.to(device) * stds[indices].to(device)
        else:
            return x * stds[indices]#

class Normalizer_LENS(torch.nn.Module):
    def __init__(self, variables, levels=True, low_levels=False, grid=False):
        super().__init__()
        self.vars = variables
        if levels:
            self.means = build_means_levels(variables, low_levels, grid)
            self.stds = build_stds_levels(variables, low_levels, grid)
        else:
            self.means = build_means(variables)
            self.stds = build_stds(variables)

    def forward(self, x):
        return normalize_tensor(x, self.means, self.stds)

    def inverse(self, x, device=None):
        return denormalize_tensor(x, self.means, self.stds, device=device)


class Denormalizer_LENS(torch.nn.Module):
    def __init__(self, variables):
        super().__init__()
        self.means = build_means(variables)
        self.stds = build_stds(variables)

    def forward(self, x):
        return denormalize_tensor(x, self.means, self.stds)

    def inverse(self, x):
        return normalize_tensor(x, self.means, self.stds)



# Normalization for ORAS5, CERA-20C
# =============================================================================
class Normalizer(torch.nn.Module):
    def __init__(self, variables):
        super().__init__()
        self.vars = variables
        self.means = None
        self.stds = None


    def fit2torch(self, data: torch.Tensor, vardim: int = None) -> None:
        """Compute the mean and std along all but the vardim dimension. 

        Args:
            data (xr.DataArray): Data to compute the normalization for.
            vardim (int, optional): Axis along the normalization should be performed. If
                dim is specified, axis will not be used. Defaults to None.
        """
        dims = tuple(d for d in range(data.dim()) if d != vardim)
        self.means = data.mean(dim=dims)
        self.stds = data.std(dim=dims, unbiased=False)

        assert len(self.means) == len(self.vars)

        return self.means, self.stds
    
    def fit2xr(self, data: xr.Dataset) -> None:
        """Compute the mean and std along all but the vardim dimension. 

        Args:
            data (xr.DataArray): Data to compute the normalization for.
            vardim (int, optional): Axis along the normalization should be performed. If
                dim is specified, axis will not be used. Defaults to None.
        """
        self.means, self.stds = torch.zeros(len(self.vars)), torch.zeros(len(self.vars))
        for i, var in enumerate(self.vars):
            self.means[i] = torch.from_numpy(data[var].fillna(0.0).mean().values)
            self.stds[i] = torch.from_numpy(data[var].fillna(0.0).std().values)

        return self.means, self.stds

    def forward(self, x):
        return normalize_tensor(x, self.means, self.stds)

    def inverse(self, x, device=None):
        return denormalize_tensor(x, self.means, self.stds, device=device)

    def to_dict(self):
        """Save variables to dict."""

        config = dict()
        for i, var in enumerate(self.vars):
            config[var] = dict()
            config[var]['mean'] = self.means[i].numpy()
            config[var]['std'] = self.stds[i].numpy()

        return config


def normalizer_from_dict(config: dict) -> Normalizer:
    """Create Normalizer object from dictionary.

    Args:
        config (dict): Dictionary with means and stds for each variable.

    Returns:
        Normalizer: Normalizer class object
    """
    normalizer = Normalizer(config.keys())
    
    means = torch.zeros(len(config.keys()))
    stds = torch.zeros(len(config.keys()))
    for i, var in enumerate(config.keys()):
        means[i] = config[var]['mean']
        stds[i] = config[var]['std']

    return normalizer