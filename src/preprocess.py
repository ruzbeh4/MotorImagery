import numpy as np
import scipy.io as sio


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