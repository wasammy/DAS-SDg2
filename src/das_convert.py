"""GTUNE seismometer velocity -> synthetic DAS strain rate (plane-wave approximation)."""
import numpy as np

def velocity_to_strain_rate(v, fs, c=3000.0, to_microstrain=True):
    v = np.asarray(v, dtype=float)
    dvdt = np.gradient(v, 1.0 / fs)   # acceleration, m/s^2
    sr = -dvdt / c                    # strain rate, 1/s
    return sr * 1e6 if to_microstrain else sr

def convert_over_speeds(v, fs, speeds=(2000, 3000, 4000, 6000, 8000)):
    return {c: velocity_to_strain_rate(v, fs, c) for c in speeds}