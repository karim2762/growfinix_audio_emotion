import json

import librosa
import numpy as np
from tqdm import tqdm

from . import config as C
from .augment import add_noise, pitch_shift


def load_audio(src) -> np.ndarray:
    y, _ = librosa.load(src, sr=C.SAMPLE_RATE, mono=True)
    return y


def preprocess_waveform(y: np.ndarray) -> np.ndarray:
    y, _ = librosa.effects.trim(y, top_db=C.TRIM_TOP_DB)
    if len(y) >= C.N_SAMPLES:
        return y[:C.N_SAMPLES].astype(np.float32)
    return np.pad(y, (0, C.N_SAMPLES - len(y))).astype(np.float32)


def log_mel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=C.SAMPLE_RATE, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH, n_mels=C.N_MELS)
    return librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)


def normalize(spec: np.ndarray) -> np.ndarray:
    return ((spec - spec.mean()) / (spec.std() + 1e-6)).astype(np.float32)


def waveform_to_input(y: np.ndarray) -> np.ndarray:
    return normalize(log_mel(preprocess_waveform(y)))


def actor_id(path) -> int:
    return int(path.stem.split("-")[-1])


def list_clips():
    classes = [e for e in C.ALL_EMOTIONS if any((C.RAW_DIR / e).glob("*.wav"))]
    if not classes:
        raise FileNotFoundError("No .wav files in data/raw/. Run: python download_data.py")
    clips = [(p, classes.index(e)) for e in classes for p in sorted((C.RAW_DIR / e).glob("*.wav"))]
    return classes, clips


def build_dataset() -> None:
    C.set_seed()
    rng = np.random.default_rng(C.SEED)
    classes, clips = list_clips()
    X, X_aug, y, actors = [], [], [], []

    for path, label in tqdm(clips, desc="Extracting features"):
        wav = preprocess_waveform(load_audio(path))
        X.append(normalize(log_mel(wav)))

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
