import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"

SEED = 42

SAMPLE_RATE = 22050
DURATION = 3.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_FRAMES = 1 + N_SAMPLES // HOP_LENGTH
TRIM_TOP_DB = 30

ALL_EMOTIONS = ["angry", "happy", "sad", "neutral"]

RAVDESS_CODES = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                 "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except ImportError:
        pass
