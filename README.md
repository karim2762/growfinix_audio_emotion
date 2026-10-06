# Task 1 – Speech Emotion Classification (angry / happy / sad)

Classify the emotion in a short speech clip. The audio is turned into a **log-Mel spectrogram** (an image of sound) and a small **CNN** reads it. Built with Python, Librosa and TensorFlow/Keras, and designed to train on a laptop CPU in well under 15 minutes.

## Quick start

```bash
cd task1_audio_emotion
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # 1. install
python download_data.py                                  # 2. download RAVDESS (~215 MB), keep angry/happy/sad
python train.py                                          # 3. extract features + train + evaluate
streamlit run app.py                                     # 4. launch the demo
```

Optional: `python download_data.py --include-neutral` (adds a 4th class), `python train.py --random-split` (see below), `jupyter lab notebooks/exploration.ipynb`.

## What gets produced

| Output | Where |
|---|---|
| Spectrograms (`X.npy`, `X_aug.npy`, `y.npy`, `actors.npy`) | `data/processed/` |
| Best model + label list | `models/best_model.keras`, `models/labels.json` |
| Accuracy/loss curves | `assets/training_curves.png` |
| Confusion matrix | `assets/confusion_matrix.png` |
| Precision/recall/F1 report | `assets/classification_report.txt` |

## How it works

```
.wav ─► load @22050 Hz ─► trim silence, pad/crop to 3 s ─► log-Mel (128×130) ─► normalise ─► CNN ─► softmax
```

- **What is a spectrogram?** Cut audio into tiny overlapping windows, take a Fourier transform of each (which frequencies are present?), and stack them: x = time, y = frequency, brightness = energy. *Mel* squeezes frequency to match human hearing; *log* matches perceived loudness. The CNN then treats it like a picture. (Also explained in `src/features.py` and the notebook.)
- **Augmentation** (training set only): noise + pitch shift (±2 semitones) are precomputed once per clip; SpecAugment time/frequency masks are applied fresh every epoch.
- **Model** (`src/model.py`): 3 × [Conv → BatchNorm → ReLU → MaxPool → Dropout] → global average pooling → Dense(64) → Dropout → Dense softmax. About 28k parameters.
- **Split**: by *actor* (16 train / 4 val / 4 test). RAVDESS actors each record every emotion equally, so this split is also class-balanced. A voice never appears in two sets, so test accuracy isn't inflated by recognising speakers. `--random-split` gives a stratified random split instead, which will look better but is optimistic.
- **Training**: EarlyStopping (val loss, restores best weights), ModelCheckpoint, ReduceLROnPlateau, class weights (applied as per-sample weights), fixed seed `42`.

## Project layout

```
task1_audio_emotion/
├── download_data.py        # Zenodo download + filter
├── train.py                # split, train, evaluate, save plots
├── app.py                  # Streamlit demo (upload or microphone)
├── src/{config,features,augment,model}.py
├── notebooks/exploration.ipynb
├── assets/  models/  data/ # outputs (data/ and *.keras are git-ignored)
└── requirements.txt
```

## Other datasets

**TESS** (Toronto Emotional Speech Set: 2 actresses, ~2,800 clips, 7 emotions) is a good alternative or addition. It requires a Kaggle/Dataverse login, so it isn't auto-downloaded. Put its clips in `data/raw/<emotion>/` with filenames ending in a speaker number (see the docstring in `download_data.py`). With only 2 speakers, speaker-independent evaluation is weak.

## Limitations

- **Acted vs. real emotion.** RAVDESS actors *perform* emotions in a studio, and the performances are exaggerated. Real anger or sadness is subtler, mixed and context-dependent. Expect a large drop on spontaneous speech (call-centre audio, voice notes).
- **Small dataset.** About 190 clips per emotion from 24 actors. Test accuracy comes from just 4 held-out speakers, so it is noisy, and a different actor split can shift results by several points. Augmentation helps but cannot replace more data.
- **Accent and demographic bias.** All speakers are North-American English actors (12 female, 12 male), saying two fixed sentences. Other accents, languages, ages, recording conditions and cultural ways of expressing emotion are not represented. The model may be systematically less accurate (or biased) for them.
- **Narrow label set.** Only 3–4 emotions; real speech is often blended (e.g. anxious + angry).
- **Not for high-stakes use.** Don't use this to judge people's emotional state, for hiring, or for surveillance.
- Microphone audio (background noise, different mics) differs from the studio recordings used to train, so confidence scores are not calibrated probabilities.

## Reproducibility notes

Seeds are fixed (Python, NumPy, TensorFlow, the actor split, augmentation). Small floating-point differences between machines/TensorFlow builds are normal. Python 3.10–3.12 recommended.
