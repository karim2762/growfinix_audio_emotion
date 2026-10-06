"""Data augmentation: more (slightly different) training examples from the same clips.

WHY: RAVDESS is tiny (~190 clips per emotion). Without augmentation a CNN
memorises the training clips instead of learning what "angry" sounds like.

Two families:
  * WAVEFORM augmentation (noise, pitch shift): done once in features.py,
    because pitch shifting is slow (we don't want to redo it every epoch).
  * SPECTROGRAM augmentation (SpecAugment): cheap, so done on-the-fly in
    train.py with fresh random masks every epoch.
"""
import librosa
import numpy as np

from . import config as C


def add_noise(y: np.ndarray, rng: np.random.Generator, snr_db_range=(15.0, 30.0)) -> np.ndarray:
    """Add white noise at a random signal-to-noise ratio (higher dB = quieter noise).

    WHY: real microphones are noisy; training on slightly noisy audio makes
    the model less fragile.
    """
    snr_db = rng.uniform(*snr_db_range)
    signal_power = np.mean(y ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape)
    return (y + noise).astype(np.float32)


def pitch_shift(y: np.ndarray, rng: np.random.Generator, max_steps: float = 2.0) -> np.ndarray:
    """Shift pitch by up to +/- max_steps semitones.

    WHY small: pitch is itself an emotion cue (angry/happy are often higher
    than sad), so big shifts could change the "true" label.
    """
    n_steps = rng.uniform(-max_steps, max_steps)
    return librosa.effects.pitch_shift(y, sr=C.SAMPLE_RATE, n_steps=n_steps).astype(np.float32)


def spec_augment(spec: np.ndarray, rng: np.random.Generator,
                 n_freq_masks: int = 2, max_freq_width: int = 16,
                 n_time_masks: int = 2, max_time_width: int = 20) -> np.ndarray:
    """SpecAugment-style masking on a (n_mels, n_frames) spectrogram.

    Blank out a few random horizontal bands (frequencies) and vertical bands
    (time spans). WHY: forces the CNN not to rely on one tiny region -
    like hiding part of a photo so the model must use the whole picture.
    We fill with 0 because the spectrogram is normalised, so 0 == "average".
    """
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
