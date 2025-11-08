import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


def correggi_ECG(ecg_pulse_pred, threshold):
    """
    Correct ECG predictions by removing peaks that are too close together.
    
    Args:
        ecg_pulse_pred: Array of predicted ECG peak times
        threshold: Minimum time difference between consecutive peaks
    
    Returns:
        Corrected ECG peak times
    """
    ECG_corr = ecg_pulse_pred.copy()
    pos = []
    ii = 1
    
    while ii < len(ecg_pulse_pred) - 1:
        if ecg_pulse_pred[ii] - ecg_pulse_pred[ii-1] < threshold:
            pos.append(ii)
            ii += 2
        else:
            ii += 1
    
    ECG_corr = np.delete(ECG_corr, pos)
    return ECG_corr


def train_pulse(X, T, A, width, freq, pulse_type='Gaussian'):
    """
    Generate a pulse train signal.
    
    Args:
        X: Input signal matrix
        T: Indices of peak positions
        A: Amplitude of pulses
        width: Width of each pulse
        freq: Sampling frequency
        pulse_type: Type of pulse ('Rectangular' or 'Gaussian')
    
    Returns:
        Pulse train signal
    """
    t = np.arange(0, X.shape[1] / freq, 1 / freq)
    T_time = t[T]
    signal = np.zeros(len(t))
    
    if pulse_type == 'Rectangular':
        for i in range(len(T_time)):
            pulse_region = (t >= T_time[i] - width/2) & (t <= T_time[i] + width/2)
            signal[pulse_region] = A
    elif pulse_type == 'Gaussian':
        for i in range(len(T_time)):
            signal += A * np.exp(-(t - T_time[i])**2 / (2 * width**2))
    
    return signal


def Non_Delay_Embedding(X, l):
    """
    Extract specific channels from multivariate series.
    
    Args:
        X: Original multivariate series (channels x time)
        l: List of channel indices to extract
    
    Returns:
        Embedded series with selected channels
    """
    return X[l, :]


def GPR_model(num_combinations, combinations, X, ECG_pulse_delay_cat):
    """
    Train GPR models for each combination of channels.
    
    Args:
        num_combinations: Number of channel combinations
        combinations: Array of channel combinations
        X: Input EEG data
        ECG_pulse_delay_cat: Target ECG pulse data
    
    Returns:
        List of trained GPR models
    """
    gprModel_pulse = []
    
    for ll in range(num_combinations):
        print(f"Training model {ll+1}/{num_combinations}...")
        
        indexes = combinations[ll, :]
        X_nemb = Non_Delay_Embedding(X, indexes)
        
        # Create GPR model with Matern kernel
        kernel = Matern(nu=1.5, length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )
        
        # Fit the model
        gpr.fit(X_nemb.T, ECG_pulse_delay_cat)
        gprModel_pulse.append(gpr)
        print(f"Model {ll+1} trained successfully")
    
    return gprModel_pulse


def GPR_prediction(num_combinations, Y, combinations, gprModel_pulse):
    """
    Make predictions using trained GPR models.
    
    Args:
        num_combinations: Number of channel combinations
        Y: Input EEG data for prediction
        combinations: Array of channel combinations
        gprModel_pulse: List of trained GPR models
    
    Returns:
        Mean prediction across all models
    """
    x_est_pulse = np.zeros((num_combinations, Y.shape[1]))
    
    for ll in range(num_combinations):
        indexes = combinations[ll, :]
        X_new = Non_Delay_Embedding(Y, indexes).T
        x_est_pulse[ll, :] = gprModel_pulse[ll].predict(X_new)
    
    X_pred_pulse = np.mean(x_est_pulse, axis=0)
    return X_pred_pulse


def Delay_Embedding(X, m):
    """
    Create delay embedding of a univariate time series.
    
    Args:
        X: Original univariate series
        m: Embedding dimension
    
    Returns:
        Delay embedded series
    """
    X = X.flatten()
    N = len(X)
    X_emb = np.zeros((m, N - m + 1))
    
    for ii in range(m):
        X_emb[ii, :] = X[ii:N - m + ii + 1]
    
    return X_emb


def combinazioni_random(N, k, n_combinazioni, seed=42):
    """
    Generate random combinations of channel indices.
    
    Args:
        N: Total number of channels
        k: Number of channels per combination
        n_combinazioni: Number of combinations to generate
        seed: Random seed for reproducibility
    
    Returns:
        Array of random channel combinations
    """
    np.random.seed(seed)
    
    if k > N:
        k = N
    
    combinazioni = np.zeros((n_combinazioni, k), dtype=int)
    for nn in range(n_combinazioni):
        combinazioni[nn, :] = np.random.choice(N, k, replace=False)
    
    return combinazioni