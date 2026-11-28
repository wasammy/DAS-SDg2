"""GTUNE seismometer -> synthetic DAS strain rate (plane-wave approximation).

Notes (from review):
- Input must have the instrument response removed first. With ObsPy that is
  st.remove_response(output="VEL") for velocity or output="ACC" for acceleration.
- DAS measures strain along the cable, so use a horizontal/radial component,
  not vertical, and record which component was used.
- Run the conversion at the RAW sample rate, before bandpass/resampling.
- The wave speed c cancels after peak normalization, so sweeping c does not
  change classifier inputs. It only matters for the amplitude check.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import detrend


def velocity_to_strain_rate(v, fs, c=3000.0, to_microstrain=True):
    """strain rate = -(1/c) * dv/dt. v in m/s, fs in Hz, c in m/s."""
    v = np.asarray(v, dtype=float)
    dvdt = np.gradient(v, 1.0 / fs)   # acceleration, m/s^2
    sr = -dvdt / c                    # strain rate, 1/s
    return sr * 1e6 if to_microstrain else sr


def acceleration_to_strain_rate(a, c=3000.0, to_microstrain=True):
    """If the data is already acceleration (m/s^2), skip the derivative: strain rate = -a/c."""
    a = np.asarray(a, dtype=float)
    sr = -a / c
    return sr * 1e6 if to_microstrain else sr


def convert_over_speeds(v, fs, speeds=(2000, 3000, 4000, 6000, 8000)):
    """Same velocity trace at a range of assumed speeds (amplitude check only)."""
    return {c: velocity_to_strain_rate(v, fs, c) for c in speeds}


def integrate_strain_rate(x, fs, axis=-1):
    """Strain rate -> strain (velocity-like: -v/c for a plane wave), so DAS matches seismometer velocity.
    Demean, trapezoid integration, linear detrend. Use at the raw rate, before preprocess_trace."""
    x = np.asarray(x, dtype=float)
    y = cumulative_trapezoid(x - x.mean(axis=axis, keepdims=True), dx=1.0 / fs, axis=axis, initial=0.0)
    return detrend(y, axis=axis, type="linear")
