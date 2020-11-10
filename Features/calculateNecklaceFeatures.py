from __future__ import division
from scipy.stats import skew, kurtosis
from numpy import mean, std, median, sqrt, hstack
import numpy as np
import scipy.fftpack
import pandas as pd
import tsfresh.feature_extraction.feature_calculators as fc


def fft_wo_offset(y, num_component=10):
    """
    return FFT (Fast Fourier transform) without offset (0 Hz) component

    Parameters
    ----------
    y: numpy array, input signal

    num_component: number of FFT component to return

    Returns
    -------
    numpy array, FFT components

    """
    N = y.shape[0] # Number of sample points
    yf = scipy.fftpack.fft(y)  # yF: discrete Fourier transform of real or complex sequence.
    amp = 2.0 / N * np.abs(yf[:int(N / 2)])
    if len(amp) < num_component+1:
        amp = hstack((amp, np.zeros(num_component+1 - len(amp))))
    return amp[1:num_component+1]


def get_feature(df, num_samples_in_FFT):
    """
    Get feature dataframe from sensor data dataframe

    Parameters
    ----------
    df: pandas DataFrame, input sensor data
    num_samples_in_FFT: number of time series samples used in FFT calculation

    Returns
    -------
    pd.DataFrame
        Description of dataframe:
        ==========          ==============================================================
        type                description

        proximity_mean      proximity features
        ...                 ...

        ambient_mean        ambient features
        ...                 ...

        leanForward_mean    leanForward features
        ...                 ...

        energy_mean         energy features
        ...                 ...

        ==========          ==============================================================
    """

    header_list = ['proximity', 'ambient', 'leanForward', 'energy']
    df_new = df[header_list]

    # ----------------------------------------------------------------
    # Generate feature names
    # ----------------------------------------------------------------
    feature_label = ["mean", "std", "max", "min", "median",
                     "skewness", "RMS", "kurtosis",
                     "quart1", "quart3", "irq",
                     "fft1", "fft2", "fft3", "fft4", "fft5",
                     "fft6", "fft7", "fft8", "fft9", "fft10",
                     "count_above_mean", "count_below_mean",
                     "first_location_of_maximum", "first_location_of_minimum",
                     "longest_strike_above_mean", "longest_strike_below_mean",
                     "number_cwt_peaks"]

    header = []
    for k in header_list:
        for feat in feature_label:
            one = k + "_" + feat
            header.extend([one])

    header.extend(["SK_prox_fft", "K_prox_fft",
                   "SK_amb_fft", "K_amb_fft",
                   "SK_lean_fft", "K_lean_fft",
                   "SK_engy_fft", "K_engy_fft",
                   "prox_amb", "prox_lean", "prox_engy",
                   "amb_lean", "amb_engy", "lean_engy"])

    prox = df_new['proximity'].values
    amb = df_new['ambient'].values
    lean = df_new['leanForward'].values
    engy = df_new['energy'].values

    R_T = df_new.values.astype(float)

    M_T = mean(R_T, axis=0)
    V_T = std(R_T, axis=0)
    MAX = R_T.max(axis=0)
    MIN = R_T.min(axis=0)
    MED = median(R_T, axis=0)
    SK_T = skew(R_T, axis=0)
    RMS_T = sqrt(mean(R_T ** 2, axis=0))
    K_T = kurtosis(R_T, axis=0)
    Q1 = np.percentile(R_T, 25, axis=0)
    Q3 = np.percentile(R_T, 75, axis=0)
    QI = Q3 - Q1

    prox_fft = fft_wo_offset(prox[:num_samples_in_FFT])
    amb_fft = fft_wo_offset(amb[:num_samples_in_FFT])
    lean_fft = fft_wo_offset(lean[:num_samples_in_FFT])
    engy_fft = fft_wo_offset(engy[:num_samples_in_FFT])

    # time series features
    count_above_mean = []
    for k in header_list:
        count_above_mean.append(fc.count_above_mean(df_new[k]))
    count_above_mean = np.array(count_above_mean)

    count_below_mean = []
    for k in header_list:
        count_below_mean.append(fc.count_below_mean(df_new[k]))
    count_below_mean = np.array(count_below_mean)

    first_location_of_maximum = []
    for k in header_list:
        first_location_of_maximum.append(fc.first_location_of_maximum(df_new[k]))
    first_location_of_maximum = np.array(first_location_of_maximum)

    first_location_of_minimum = []
    for k in header_list:
        first_location_of_minimum.append(fc.first_location_of_minimum(df_new[k]))
    first_location_of_minimum = np.array(first_location_of_minimum)

    longest_strike_above_mean = []
    for k in header_list:
        longest_strike_above_mean.append(fc.longest_strike_above_mean(df_new[k]))
    longest_strike_above_mean = np.array(longest_strike_above_mean)

    longest_strike_below_mean = []
    for k in header_list:
        longest_strike_below_mean.append(fc.longest_strike_below_mean(df_new[k]))
    longest_strike_below_mean = np.array(longest_strike_below_mean)

    number_cwt_peaks = []
    for k in header_list:
        number_cwt_peaks.append(fc.number_cwt_peaks(df_new[k], 10))
    number_cwt_peaks = np.array(number_cwt_peaks)

    SK_prox_fft = skew(prox_fft)
    K_prox_fft = kurtosis(prox_fft)
    SK_amb_fft = skew(amb_fft)
    K_amb_fft = kurtosis(amb_fft)
    SK_lean_fft = skew(lean_fft)
    K_lean_fft = kurtosis(lean_fft)
    SK_engy_fft = skew(engy_fft)
    K_engy_fft = kurtosis(engy_fft)

    COV_M = np.cov(R_T.T)
    COV = np.array([COV_M[0, 1], COV_M[0, 2], COV_M[0, 3], COV_M[1, 2], COV_M[1, 3], COV_M[2, 3]])

    H_T = hstack((M_T, V_T, MAX, MIN, MED, SK_T, RMS_T, K_T, Q1, Q3, QI,
                  prox_fft, amb_fft, lean_fft, engy_fft,
                  count_above_mean, count_below_mean,
                  first_location_of_maximum, first_location_of_minimum,
                  longest_strike_above_mean, longest_strike_below_mean,
                  number_cwt_peaks,
                  SK_prox_fft, K_prox_fft, SK_amb_fft, K_amb_fft, SK_lean_fft, K_lean_fft, SK_engy_fft, K_engy_fft,
                  COV))

    feat_df = pd.DataFrame(data=H_T[np.newaxis, :], columns=header)

    return feat_df
