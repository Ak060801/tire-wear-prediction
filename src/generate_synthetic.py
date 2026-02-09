#!/usr/bin/env python3
"""
Deterministic synthetic telemetry generator.
Each sample represents a current lap window of length `seq_len` timesteps and
contains multiple sensors (tire_temp, pressure, speed, throttle, brake, rpm).
Label y=1 if predicted wear at lap + horizon (12 laps) > threshold.
"""
import argparse
import numpy as np
import os

def generate(n_samples=10000, seq_len=50, sensors=6, horizon=12, random_seed=42):
    rng = np.random.default_rng(random_seed)

    # create a deterministic base "wear slope" per sample in range [0.01, 0.15]
    base_slope = rng.uniform(0.01, 0.15, size=(n_samples,1))

    # per-sensor influence factors (tire_temp most predictive)
    sensor_weights = np.array([1.0, 0.4, 0.2, 0.1, 0.05, 0.02])[:sensors]

    # create sequences: for each sample, sensor readings over seq_len timesteps
    data = np.zeros((n_samples, seq_len, sensors), dtype=np.float32)

    lap_index = rng.integers(5, 40, size=(n_samples,1))  # current lap index (affects baseline wear)

    for i in range(n_samples):
        slope = base_slope[i,0]
        lap0 = lap_index[i,0]
        t = np.arange(seq_len)

        # tire_temp: baseline increases with lap and slope, plus small periodic spikes (corners)
        tire_temp = 60 + 0.8*lap0 + slope*100*t/seq_len + 3.0*np.sin(2*np.pi*t/10) + rng.normal(0, 0.5, size=seq_len)
        # pressure: slowly decreases with wear
        pressure = 22 - 0.1*lap0 - slope*30*t/seq_len + rng.normal(0, 0.05, size=seq_len)
        # speed: varies per timestep with noise
        speed = 200 - 0.5*lap0 + 5*np.cos(2*np.pi*t/20) + rng.normal(0, 2, size=seq_len)
        # throttle, brake, rpm: noisy signals
        throttle = 0.6 + 0.1*np.sin(2*np.pi*t/15) + rng.normal(0, 0.02, size=seq_len)
        brake = 0.1 + 0.05*np.maximum(0, np.sin(2*np.pi*t/10)) + rng.normal(0, 0.01, size=seq_len)
        rpm = 12000 - 20*lap0 + rng.normal(0, 50, size=seq_len)

        sensors_arr = np.vstack([tire_temp, pressure, speed, throttle, brake, rpm]).T[:, :sensors]
        data[i] = sensors_arr

    # Now define a ground-truth wear function: current_wear + horizon_increment
    # current_wear is proportional to weighted sum of last-window slopes + lap index * small factor
    last_window_mean = data.mean(axis=1)
    # compute per-sample slope estimator for tire_temp-like sensor
    tire_temp_series = data[:, :, 0]  # (n_samples, seq_len)
    idx = np.arange(seq_len)
    slopes = np.sum((idx - idx.mean()) * (tire_temp_series - tire_temp_series.mean(axis=1, keepdims=True)), axis=1) / np.sum((idx - idx.mean())**2)

    current_wear = 0.02*lap_index.flatten() + 0.5*slopes
    # horizon wear increases by slope * horizon_factor
    horizon_factor = 0.12  # converts slope to expected wear per lap
    future_wear = current_wear + slopes * horizon * horizon_factor

    # label: worn within horizon if future_wear > threshold
    threshold = np.percentile(future_wear, 75) * 0.9  # choose threshold so roughly 25% positive rate
    labels = (future_wear > threshold).astype(int)

    # Save everything in a dict
    out = {
        'data': data.astype(np.float32),
        'labels': labels.astype(np.int8),
        'lap_index': lap_index.flatten().astype(np.int16),
        'slopes': slopes.astype(np.float32),
        'future_wear': future_wear.astype(np.float32)
    }
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/raw/f1_telemetry.npy')
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    d = generate(n_samples=args.n, random_seed=args.seed)
    np.save(args.out, d)
    print(f"Saved synthetic telemetry to {args.out} (samples: {args.n})")
