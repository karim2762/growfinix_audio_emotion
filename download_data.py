"""Download RAVDESS (speech) from Zenodo and keep only angry / happy / sad.

Usage:
    python download_data.py                    # angry, happy, sad
    python download_data.py --include-neutral  # + neutral

Result: data/raw/<emotion>/<original-filename>.wav

RAVDESS filename = 7 numbers separated by '-':
    modality-vocalChannel-EMOTION-intensity-statement-repetition-ACTOR
    e.g. 03-01-05-01-02-01-12.wav  -> emotion 05 (angry), actor 12
The actor id matters: we use it later for a speaker-independent split.

Alternative dataset: TESS (Toronto Emotional Speech Set: 2 actresses, ~2800 clips,
7 emotions). It needs a Kaggle/Dataverse login, so it is not auto-downloaded here.
To use it, copy the angry/happy/sad clips into data/raw/<emotion>/ and rename each
file to end in a speaker number, e.g. "OAF_back_angry.wav" -> "OAF_back-1.wav"
(OAF=1, YAF=2). Heads-up: with only 2 speakers a speaker-independent split is weak.
"""
import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from src import config as C

URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
ZIP_PATH = C.DATA_DIR / "Audio_Speech_Actors_01-24.zip"


def download(url: str, dest: Path) -> None:
    """Stream the file to disk (so we never hold ~215 MB in RAM) with a progress bar."""
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")  # write to .part so a crash can't leave a corrupt .zip
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def extract_filtered(zip_path: Path, keep: set) -> dict:
    """Copy only the wanted emotions straight out of the zip (no need to unzip everything)."""
    counts = {e: 0 for e in keep}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            p = Path(name)
            if p.suffix.lower() != ".wav" or p.name.startswith("."):
                continue
            parts = p.stem.split("-")
            if len(parts) != 7:
                continue
            emotion = C.RAVDESS_CODES.get(parts[2])
            if emotion not in keep:
                continue
            out = C.RAW_DIR / emotion / p.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(name))
            counts[emotion] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-neutral", action="store_true", help="also keep 'neutral'")
    ap.add_argument("--keep-zip", action="store_true", help="don't delete the zip afterwards")
    args = ap.parse_args()

    keep = {"angry", "happy", "sad"} | ({"neutral"} if args.include_neutral else set())
    download(URL, ZIP_PATH)
    counts = extract_filtered(ZIP_PATH, keep)
    print("Clips kept:", counts)
    if not args.keep_zip:
        ZIP_PATH.unlink()
        print("Removed zip (use --keep-zip to keep it).")


if __name__ == "__main__":
    main()
