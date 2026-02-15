import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from scipy.signal import welch
import matplotlib.pyplot as plt


def load_and_window_data(filepath, fs=100, window_sec=(0.5, 3.5)):
    """
    Loads BCI .mat dataset, extracts 'pos' and windows the continuous signal.
    """
    mat_data = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)

    # standard BCI IV format structure.
    cnt = mat_data['cnt']  # Continuous signal
    mrk = mat_data['mrk']  # Marker info
    
    pos = mrk.pos  # Starting point of each window
    y = mrk.y      # Labels for left hand vs foot
    
    # Calculate window sizes in samples
    start_sample = int(window_sec[0] * fs)
    end_sample = int(window_sec[1] * fs)
    window_length = end_sample - start_sample
    
    n_trials = len(pos)
    n_channels = cnt.shape[1]
    
    # array for windowed data: (trials, channels, time_samples)
    X = np.zeros((n_trials, n_channels, window_length))
    
    for i, start_pos in enumerate(pos):
        X[i, :, :] = cnt[start_pos + start_sample : start_pos + end_sample, :].T
        
    # Keep only the Left Hand and Foot
    valid_trials = (y == 1) | (y == 2)
    X = X[valid_trials]
    y = y[valid_trials]
    
    return X, y


def split_data(X, y):
    """
    Splits data into 75% train and 25% test.
    """
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def apply_bandpass_filter(X, fs=100, lowcut=8.0, highcut=30.0, order=4):
    """
    Applies a Butterworth bandpass filter to extract Mu and Beta bands.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter along the time axis (axis=-1)
    X_filtered = filtfilt(b, a, X, axis=-1)
    return X_filtered

def plot_psd_comparison(raw_X, filtered_X, fs=100, channel_idx=0):
    """
    Plots the Power Spectral Density to show the effect of the bandpass filter.
    """
    # Calculate PSD for raw and filtered data for a single trial/channel
    freqs_raw, psd_raw = welch(raw_X[0, channel_idx, :], fs, nperseg=fs*2)
    freqs_filt, psd_filt = welch(filtered_X[0, channel_idx, :], fs, nperseg=fs*2)

    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs_raw, psd_raw, label='Raw Signal', alpha=0.7)
    plt.semilogy(freqs_filt, psd_filt, label='Filtered (8-30 Hz)', color='orange', linewidth=2)
    
    # Highlight the passband (Mu + Beta)
    plt.axvspan(8, 30, color='green', alpha=0.1, label='Target Passband')
    
    plt.title(f'Power Spectral Density Comparison (Channel {channel_idx})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()