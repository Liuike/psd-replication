"""Diagnostics for PSD latent geometry and periodic MuJoCo trajectories."""

from __future__ import annotations

import math

import numpy as np


def geometry(z, z_next, z_L, L: int) -> dict[str, object]:
    """Measure Equation 5/6 geometry for aligned encoded state batches.

    Each input has shape (batch, latent_dim). z_next must encode s[t+1]
    and z_L must encode s[t+L] for the same transitions.
    """
    if not isinstance(L, (int, np.integer)) or L < 1:
        raise ValueError("L must be a positive integer")

    z = np.asarray(z, dtype=np.float64)
    z_next = np.asarray(z_next, dtype=np.float64)
    z_L = np.asarray(z_L, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError("encoded states must have shape (nonzero batch, latent_dim)")
    if z_next.shape != z.shape or z_L.shape != z.shape:
        raise ValueError("z, z_next, and z_L must have the same shape")
    if not (np.isfinite(z).all() and np.isfinite(z_next).all() and np.isfinite(z_L).all()):
        raise ValueError("encoded states must contain only finite values")

    step_distances = np.linalg.norm(z_next - z, axis=1)
    lag_distances = np.linalg.norm(z_L - z, axis=1)
    antipodal_residuals = np.linalg.norm(z_L + z, axis=1)
    step_target = L * math.sin(math.pi / (2 * L))
    lag_target = float(L)

    def summarize(values: np.ndarray, target: float) -> dict[str, float]:
        average = float(values.mean())
        return {
            "mean": average,
            "std": float(values.std()),
            "target": float(target),
            "absolute_relative_error": abs(average - target) / target,
            "mean_absolute_relative_error": float((np.abs(values - target) / target).mean()),
        }

    return {
        "L": int(L),
        "step_distance": summarize(step_distances, step_target),
        "lag_L_distance": summarize(lag_distances, lag_target),
        "antipodal_residual": {
            "mean": float(antipodal_residuals.mean()),
            "std": float(antipodal_residuals.std()),
            "normalized_mean": float(antipodal_residuals.mean() / L),
        },
    }


def spectrum(states, random_mean=None, random_std=None) -> dict[str, object]:
    """Estimate periods in a state trajectory using PCA and a real FFT.

    The first PCA component is computed from the centered trajectory. If
    random_mean and random_std are provided, each state dimension is
    normalized with those random-policy statistics before PCA, as in the
    paper's trajectory analysis. Constant or invalid random-policy standard
    deviations are rejected rather than silently distorting the spectrum.

    Returned period estimates are in environment steps. Peak magnitudes are
    absolute, unnormalized FFT magnitudes, suitable for ranking within one
    trajectory.
    """
    trajectory = np.asarray(states, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] < 4 or trajectory.shape[1] == 0:
        raise ValueError("states must have shape (at least 4 timesteps, state_dim)")
    if not np.isfinite(trajectory).all():
        raise ValueError("states must contain only finite values")
    if (random_mean is None) != (random_std is None):
        raise ValueError("random_mean and random_std must be provided together")

    if random_mean is not None:
        random_mean = np.asarray(random_mean, dtype=np.float64)
        random_std = np.asarray(random_std, dtype=np.float64)
        if random_mean.shape != (trajectory.shape[1],) or random_std.shape != random_mean.shape:
            raise ValueError("random_mean and random_std must have shape (state_dim,)")
        if not (np.isfinite(random_mean).all() and np.isfinite(random_std).all()):
            raise ValueError("random-policy statistics must be finite")
        if np.any(random_std <= 0):
            raise ValueError("random_std must be positive for every state dimension")
        trajectory = (trajectory - random_mean) / random_std

    centered = trajectory - trajectory.mean(axis=0, keepdims=True)
    if np.linalg.norm(centered) <= np.finfo(np.float64).eps:
        raise ValueError("trajectory has no variation after centering")
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ right_vectors[0]

    amplitudes = np.abs(np.fft.rfft(projection))
    frequencies = np.fft.rfftfreq(projection.shape[0], d=1.0)
    amplitudes[0] = 0.0
    peak_bins = [
        idx for idx in range(1, len(amplitudes) - 1)
        if amplitudes[idx] >= amplitudes[idx - 1]
        and amplitudes[idx] > amplitudes[idx + 1]
    ]
    if len(amplitudes) > 1 and amplitudes[-1] >= amplitudes[-2]:
        peak_bins.append(len(amplitudes) - 1)
    if not peak_bins:
        peak_bins = list(range(1, len(amplitudes)))
    peak_bins.sort(key=lambda idx: amplitudes[idx], reverse=True)

    peaks = [
        {
            "period_steps": float(1.0 / frequencies[idx]),
            "frequency_cycles_per_step": float(frequencies[idx]),
            "magnitude": float(amplitudes[idx]),
            "relative_magnitude": float(amplitudes[idx] / amplitudes[peak_bins[0]]),
        }
        for idx in peak_bins[:5]
    ]
    return {
        "timesteps": int(projection.shape[0]),
        "pca_explained_variance_ratio": float(
            singular_values[0] ** 2 / np.sum(singular_values**2)
        ),
        "dominant_period_steps": peaks[0]["period_steps"],
        "top_peaks": peaks,
    }
