''' Collection of metrics.

@Author  :   Jakob Schlör 
@Time    :   2022/10/18 15:29:31
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import numpy as np
import pandas as pd
import xarray as xr
import torch
import scipy.stats as stats
from scipy.special import erf
from scipy.fft import fft, fftfreq


def power_spectrum(data):
    """Compute power spectrum.

    Args:
        data (np.ndarray): Data of dimension (n_feat, n_time) 

    Returns:
        xf (np.ndarray): Frequencies of dimension (n_time//2) 
        yf (np.ndarray): Power spectrum of dimension (n_feat, n_time//2) 
    """
    n_feat, n_time = data.shape
    yf = []
    for i in range(n_feat):
        yf.append(fft(data[i,:]))

    xf = fftfreq(n_time, 1)[:n_time//2]
    yf = 2./n_time * np.abs(yf)[:, :n_time//2]
    return xf, yf 


def correlation_coefficient(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Correlation coefficient.

    Args:
        x (np.ndarray): Ground truth data of size (n_samples, n_features)
        x_hat (np.ndarray): Prediction of size (n_samples, n_features)

    Returns:
        cc (np.ndarray): CC of size (n_features) 
    """
    cc = np.array(
        [stats.pearsonr(x[:, i], x_hat[:, i])[0]
         for i in range(x.shape[1])]
    )
    return cc


def crps_gaussian(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """CRPS for Gaussian distribution.

    Args:
        x (np.ndarray): Ground truth data of size (n_samples, n_features)
        mu (np.ndarray): Mean of size (n_samples, n_features)
        std (np.ndarray): Standard deviation of size (n_samples, n_features)

    Returns:
        crps (np.ndarray): CRPS of size (n_samples, n_features) 
    """
    sqrtPi = np.sqrt(np.pi)
    z = (x - mu) / std 
    phi = np.exp(-z ** 2 / 2) / (np.sqrt(2) * sqrtPi) #standard normal pdf
    crps = std * (z * erf(z / np.sqrt(2)) + 2 * phi - 1 / sqrtPi) #crps as per Gneiting et al 2005
    return crps 



def verification_metrics_per_gridpoint(target: xr.Dataset, 
                                       frcst_mean: xr.Dataset,
                                       frcst_std: xr.Dataset=None) -> dict:
    verification_metrics = {}
    # Point metrics
    # MSE
    mse = ((target - frcst_mean)**2).mean(dim='time', skipna=True)
    verification_metrics['mse'] = mse

    # RMSE skill score
    skill = 1 - np.sqrt(mse) / target.std(dim='time', skipna=True)
    verification_metrics['rmse_skill'] = skill

    # Correlation coefficient
    verification_metrics['cc'] = xr.merge([
            xr.corr(target[var], frcst_mean[var], dim='time') 
            for var in target.data_vars
    ])

    # Ensemble metrics
    if frcst_std is not None:
        # CRPS
        crps = crps_gaussian(target, frcst_mean, frcst_std)
        verification_metrics['crps'] = crps.mean(dim='time', skipna=True)
        verification_metrics['crps_norm'] = crps.mean(dim='time', skipna=True) / target.std(dim='time', skipna=True)

    return verification_metrics


def anomaly_correlation_coefficient_old(forecasts, observed, lag, cfg, climatology=None, nino_index=False, keep_spatial_coords=False,):
    """
    Calculate the Anomaly Correlation Coefficient (ACC) between forecast and observed data with a specified lag.

    Parameters:
    forecasts (list of numpy.ndarray): List of forecasted anomalies with shape [1, 20, 52, 116].
    observed (list of numpy.ndarray): List of observed anomalies with shape [1, 20, 52, 116].
    lag (int): The lag for which the correlation should be calculated.
    climatology (numpy.ndarray): Climatology of the observed data with shape [12, 52, 116].

    Returns:
    list of float: List of Anomaly Correlation Coefficients for each forecast-observed pair with the specified lag.
    """
    acc_list = []
    for forecast, obs in zip(forecasts, observed):

        forecast = forecast[0]
        obs = obs[0]

        forecast = np.asarray(forecast.cpu())
        obs = np.asarray(obs.cpu())

        # Apply nino index region if specified
        if nino_index == "nino_3":
            forecast = forecast[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_3_lon"][0]:cfg["nino_3_lon"][1]]
            obs = obs[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_3_lon"][0]:cfg["nino_3_lon"][1]]
        if nino_index == "nino_34":
            forecast = forecast[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
            obs = obs[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
        elif nino_index == "nino_4":
            forecast = forecast[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]
            obs = obs[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]

        if isinstance(lag, int):
            # Apply lag
            forecast = forecast[:, lag, :, :]
            obs = obs[:, lag, :, :]
            #context = context[0, lag]
        else:
            pass
        
        # Calculate the anomalies (deviations from the mean) along the specified axes
        #forecast_anomalies = forecast - climatology[context, :, :]
        #observed_anomalies = obs - climatology[context, :, :]


        if keep_spatial_coords:
            # Compute the numerator (covariance)
            numerator = np.sum(forecast * obs, axis=0)

            # Compute the denominator (standard deviations)
            denominator = np.sqrt(
                np.sum(forecast**2, axis=0) * np.sum(obs**2, axis=0)
            )

            # Compute the Anomaly Correlation Coefficient
            acc = numerator / denominator

            # Append the scalar value to the result list
            acc_list.append(acc)
        else:
            # Compute the numerator (covariance)
            numerator = np.sum(forecast * obs)

            # Compute the denominator (standard deviations)
            denominator = np.sqrt(
                np.sum(forecast**2) * np.sum(obs**2)
            )

            # Compute the Anomaly Correlation Coefficient
            acc = numerator / denominator

            # Append the scalar value to the result list
            acc_list.append(acc)

    return acc_list

def anomaly_correlation_coefficient(forecasts, observed, lag, cfg, climatology=None, nino_index=False, keep_spatial_coords=False):
    """
    Calculate the Anomaly Correlation Coefficient (ACC) between forecast and observed data with a specified lag,
    concatenating all elements of the lists on the first dimension before calculating the ACC.

    Parameters:
    forecasts (list of numpy.ndarray): List of forecasted anomalies with shape [1, 20, 52, 116].
    observed (list of numpy.ndarray): List of observed anomalies with shape [1, 20, 52, 116].
    lag (int): The lag for which the correlation should be calculated.
    climatology (numpy.ndarray): Climatology of the observed data with shape [12, 52, 116].

    Returns:
    float: The Anomaly Correlation Coefficient calculated on the concatenated data.
    """
    # Concatenate all elements in the lists along the first dimension
    forecast_concat = np.concatenate([np.asarray(f[0].cpu()) for f in forecasts], axis=0)
    observed_concat = np.concatenate([np.asarray(o[0].cpu()) for o in observed], axis=0)

    # Apply nino index region if specified
    if nino_index == "nino_3":
        forecast_concat = forecast_concat[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_3_lon"][0]:cfg["nino_3_lon"][1]]
        observed_concat = observed_concat[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_3_lon"][0]:cfg["nino_3_lon"][1]]
    elif nino_index == "nino_34":
        forecast_concat = forecast_concat[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
        observed_concat = observed_concat[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
    elif nino_index == "nino_4":
        forecast_concat = forecast_concat[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]
        observed_concat = observed_concat[:, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]

    if isinstance(lag, int):
        # Apply lag
        forecast_concat = forecast_concat[:, lag, :, :]
        observed_concat = observed_concat[:, lag, :, :]

    # Calculate anomalies and ACC
    if keep_spatial_coords:
        # Compute the numerator (covariance) along the concatenated data
        numerator = np.sum(forecast_concat * observed_concat, axis=0)

        # Compute the denominator (standard deviations)
        denominator = np.sqrt(
            np.sum(forecast_concat**2, axis=0) * np.sum(observed_concat**2, axis=0)
        )

        # Compute the Anomaly Correlation Coefficient
        acc = numerator / denominator
    else:
        # Compute the numerator (covariance)
        numerator = np.sum(forecast_concat * observed_concat)

        # Compute the denominator (standard deviations)
        denominator = np.sqrt(
            np.sum(forecast_concat**2) * np.sum(observed_concat**2)
        )

        # Compute the Anomaly Correlation Coefficient
        acc = numerator / denominator

    return acc


def anomaly_correlation_coefficient_all_depths(forecasts, observed, lag, cfg, climatology=None, nino_index=False, keep_spatial_coords=False,):
    """
    Calculate the Anomaly Correlation Coefficient (ACC) between forecast and observed data with a specified lag.

    Parameters:
    forecasts (list of numpy.ndarray): List of forecasted anomalies with shape [1, 20, 52, 116].
    observed (list of numpy.ndarray): List of observed anomalies with shape [1, 20, 52, 116].
    lag (int): The lag for which the correlation should be calculated.
    climatology (numpy.ndarray): Climatology of the observed data with shape [12, 52, 116].

    Returns:
    list of float: List of Anomaly Correlation Coefficients for each forecast-observed pair with the specified lag.
    """
    # Concatenate all elements in the lists along the first dimension
    forecast_concat = np.concatenate([np.asarray(f[0].cpu()) for f in forecasts], axis=0)
    observed_concat = np.concatenate([np.asarray(o[0].cpu()) for o in observed], axis=0)

    # Apply nino index region if specified
    if nino_index == "nino_3":
        forecast_concat = forecast_concat[:, :, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_3_lon"][0]:cfg["nino_3_lon"][1]]
        observed_concat = observed_concat[:, :, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_3_lon"][0]:cfg["nino_3_lon"][1]]
    elif nino_index == "nino_34":
        forecast_concat = forecast_concat[:, :, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
        observed_concat = observed_concat[:, :, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_34_lon"][0]:cfg["nino_34_lon"][1]]
    elif nino_index == "nino_4":
        forecast_concat = forecast_concat[:, :, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]
        observed_concat = observed_concat[:, :, :, cfg["nino_3_lat"][0]:cfg["nino_3_lat"][1], cfg["nino_4_lon"][0]:cfg["nino_4_lon"][1]]

    if isinstance(lag, int):
        # Apply lag
        forecast_concat = forecast_concat[:, lag, :, :]
        observed_concat = observed_concat[:, lag, :, :]

    # Calculate anomalies and ACC
    if keep_spatial_coords:
        # Compute the numerator (covariance) along the concatenated data
        numerator = np.sum(forecast_concat * observed_concat, axis=0)

        # Compute the denominator (standard deviations)
        denominator = np.sqrt(
            np.sum(forecast_concat**2, axis=0) * np.sum(observed_concat**2, axis=0)
        )

        # Compute the Anomaly Correlation Coefficient
        acc = numerator / denominator
    else:
        # Compute the numerator (covariance)
        numerator = np.sum(forecast_concat * observed_concat)

        # Compute the denominator (standard deviations)
        denominator = np.sqrt(
            np.sum(forecast_concat**2) * np.sum(observed_concat**2)
        )

        # Compute the Anomaly Correlation Coefficient
        acc = numerator / denominator

    return acc