"""Central settings. Every other file imports from here.

WHY: keeping constants in ONE place means training and the app can never
disagree (e.g. a different sample rate would silently break predictions).
"""
import os
import random
from pathlib import Path

import numpy as np

# ---- Paths (pathlib => works on Windows/Mac/Linux) --------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # data/raw/<emotion>/*.wav
PROCESSED_DIR = DATA_DIR / "processed"  # .npy spectrograms
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"            # plots + report

# ---- Reproducibility --------------------------------------------------------
SEED = 42

# ---- Audio / spectrogram settings ------------------------------------------
SAMPLE_RATE = 22050                      # Hz; plenty for speech, cheap on CPU
DURATION = 3.0                           # seconds; fixed so the CNN gets a fixed-size input
N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 66150 samples
N_MELS = 128                             # frequency bands (image height)
N_FFT = 2048                             # window size for each FFT
HOP_LENGTH = 512                         # step between windows
N_FRAMES = 1 + N_SAMPLES // HOP_LENGTH   # 130 time steps (image width)
TRIM_TOP_DB = 30                         # silence trimming threshold

# ---- Labels -----------------------------------------------------------------
ALL_EMOTIONS = ["angry", "happy", "sad", "neutral"]  # neutral is optional
# RAVDESS filename field 3 -> emotion name
RAVDESS_CODES = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                 "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}


def set_seed(seed: int = SEED) -> None:
    """Fix every random number generator we use so runs are repeatable.

    Note: TensorFlow on CPU is *mostly* deterministic; tiny float differences
    between machines are normal.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras
        keras.utils.set_random_seed(seed)  # seeds python, numpy and TF together
    except ImportError:
        pass
