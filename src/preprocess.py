import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

def bandpass(x, fs, fmin=0.5, fmax=10, order=4):
    x = np.asarray(x, dtype=float)
    x= x-np.mean(x) #demeaning
    sos = butter(order, [fmin, fmax], btype='bandpass', fs=fs, output='sos')
    x = sosfiltfilt(sos, x)
    return x

def resample(x, fs, target=20.0):
    if fs == target:
        return x,
    if fs <target:
        raise ValueError(f"sampling rate {fs} Hz is below the target {target} Hz")
    return resample_poly(x, target, fs)






