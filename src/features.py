"""Turn .wav files into normalised log-Mel spectrograms and save them as .npy.

----------------------------------------------------------------------------
WHAT IS A SPECTROGRAM? (the idea in 5 lines)
  * A waveform is loudness over time - hard for a model to "see" pitch/timbre in.
  * Chop the audio into tiny overlapping windows (~93 ms here) and run a Fourier
    transform on each: that tells you HOW MUCH of each frequency is present.
  * Stack the windows side by side -> an image: x = time, y = frequency,
    brightness = energy. That is a spectrogram.
  * "Mel" squeezes the frequency axis to match human hearing (we hear low pitches
    in finer detail than high ones). "Log" (decibels) matches how we perceive
    loudness. Result: a 128 x 130 image a CNN can read like any other picture.
----------------------------------------------------------------------------

Run:  python -m src.features
"""
import json

import librosa
import numpy as np
from tqdm import tqdm

from . import config as C
from .augment import add_noise, pitch_shift


def load_audio(src) -> np.ndarray:
    """Load a path or file-like object as mono float32 at 22050 Hz.

    WHY resample: every file must share one sample rate or the "pixels" of our
    spectrogram would mean different frequencies for different files.
    """
    y, _ = librosa.load(src, sr=C.SAMPLE_RATE, mono=True)
    return y


def preprocess_waveform(y: np.ndarray) -> np.ndarray:
    """Trim leading/trailing silence, then pad/crop to exactly 3 seconds.

    WHY: a CNN needs equal-sized inputs. Trimming first stops us from feeding
    the model 0.5 s of silence that carries no emotion.
    """
    y, _ = librosa.effects.trim(y, top_db=C.TRIM_TOP_DB)
    if len(y) >= C.N_SAMPLES:
        return y[:C.N_SAMPLES].astype(np.float32)
    return np.pad(y, (0, C.N_SAMPLES - len(y))).astype(np.float32)  # pad with silence


def log_mel(y: np.ndarray) -> np.ndarray:
    """Waveform (66150,) -> log-Mel spectrogram in dB, shape (128, 130)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=C.SAMPLE_RATE, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH, n_mels=C.N_MELS)
    return librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)


def normalize(spec: np.ndarray) -> np.ndarray:
    """Standardise each spectrogram to mean 0, std 1.

    WHY per-clip: removes "how loud was the recording / how close was the mic",
    which would otherwise be a distracting shortcut. Same function is used by
    the app so training and prediction match exactly.
    """
    return ((spec - spec.mean()) / (spec.std() + 1e-6)).astype(np.float32)


def waveform_to_input(y: np.ndarray) -> np.ndarray:
    """Raw waveform -> model-ready (128, 130) array. Used by the Streamlit app."""
    return normalize(log_mel(preprocess_waveform(y)))


def actor_id(path) -> int:
    """RAVDESS: the last '-' token of the filename is the actor (speaker) id."""
    return int(path.stem.split("-")[-1])


def list_clips():
    """Find clips in data/raw/<emotion>/. Class order follows C.ALL_EMOTIONS."""
    classes = [e for e in C.ALL_EMOTIONS if any((C.RAW_DIR / e).glob("*.wav"))]
    if not classes:
        raise FileNotFoundError("No .wav files in data/raw/. Run: python download_data.py")
    clips = [(p, classes.index(e)) for e in classes for p in sorted((C.RAW_DIR / e).glob("*.wav"))]
    return classes, clips


def build_dataset() -> None:
    """Compute clean + augmented spectrograms for every clip and save as .npy."""
    C.set_seed()
    rng = np.random.default_rng(C.SEED)
    classes, clips = list_clips()
    X, X_aug, y, actors = [], [], [], []

    for path, label in tqdm(clips, desc="Extracting features"):
        wav = preprocess_waveform(load_audio(path))
        X.append(normalize(log_mel(wav)))
        # One augmented copy per clip (pitch shift + noise). Row i of X_aug is the
        # twin of row i of X, so train.py can include a twin ONLY if its original
        # is in the training set - this prevents leakage into val/test.
        X_aug.append(normalize(log_mel(add_noise(pitch_shift(wav, rng), rng))))
        y.append(label)
        actors.append(actor_id(path))

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(C.PROCESSED_DIR / "X.npy", np.stack(X))
    np.save(C.PROCESSED_DIR / "X_aug.npy", np.stack(X_aug))
    np.save(C.PROCESSED_DIR / "y.npy", np.array(y, dtype=np.int64))
    np.save(C.PROCESSED_DIR / "actors.npy", np.array(actors, dtype=np.int64))
    (C.PROCESSED_DIR / "labels.json").write_text(json.dumps(classes))
    print(f"Saved {len(y)} clips (+{len(y)} augmented twins) for classes {classes} -> {C.PROCESSED_DIR}")


if __name__ == "__main__":
    build_dataset()
