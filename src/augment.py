import librosa
import numpy as np

from . import config as C


def add_noise(y: np.ndarray, rng: np.random.Generator, snr_db_range=(15.0, 30.0)) -> np.ndarray:
    snr_db = rng.uniform(*snr_db_range)
    signal_power = np.mean(y ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape)
    return (y + noise).astype(np.float32)


def pitch_shift(y: np.ndarray, rng: np.random.Generator, max_steps: float = 2.0) -> np.ndarray:
    n_steps = rng.uniform(-max_steps, max_steps)
    return librosa.effects.pitch_shift(y, sr=C.SAMPLE_RATE, n_steps=n_steps).astype(np.float32)


def spec_augment(spec: np.ndarray, rng: np.random.Generator,
                 n_freq_masks: int = 2, max_freq_width: int = 16,
                 n_time_masks: int = 2, max_time_width: int = 20) -> np.ndarray:
    out = spec.copy()
    n_mels, n_frames = out.shape
    for _ in range(n_freq_masks):
        w = int(rng.integers(0, max_freq_width + 1))
        f0 = int(rng.integers(0, n_mels - w + 1))
        out[f0:f0 + w, :] = 0.0
    for _ in range(n_time_masks):
        w = int(rng.integers(0, max_time_width + 1))
        t0 = int(rng.integers(0, n_frames - w + 1))
        out[:, t0:t0 + w] = 0.0
    return out
