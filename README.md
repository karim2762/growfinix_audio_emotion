# Speech Emotion Classification (angry / happy / sad)

Classifies the emotion in a short speech clip. The audio is converted to a log-Mel spectrogram and a small CNN predicts the emotion. Built with Python, Librosa and TensorFlow/Keras. It trains on a laptop CPU in a few minutes.

## Setup

```bash
cd growfinix_audio_emotion
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py          # RAVDESS (~215 MB), keeps angry/happy/sad
python train.py
streamlit run app.py
```

Options: `python download_data.py --include-neutral` adds a 4th class, and `python train.py --random-split` uses a random split instead of splitting by speaker.

## Output

| Output | Where |
|---|---|
| Spectrograms and labels | `data/processed/` |
| Best model and label list | `models/best_model.keras`, `models/labels.json` |
| Accuracy/loss curves | `assets/training_curves.png` |
| Confusion matrix | `assets/confusion_matrix.png` |
| Precision/recall/F1 report | `assets/classification_report.txt` |

## How it works

```
.wav -> load at 22050 Hz -> trim silence, pad/crop to 3 s -> log-Mel (128x130) -> normalise -> CNN -> softmax
```

- Augmentation (training set only): noise and a pitch shift of up to 2 semitones are computed once per clip. SpecAugment masks are applied every epoch.
- Model (`src/model.py`): 3 conv blocks (Conv, BatchNorm, ReLU, MaxPool, Dropout), global average pooling, Dense(64), Dropout, softmax output. Roughly 28k parameters.
- Split: by actor, 16 train / 4 val / 4 test, so no voice appears in more than one set. `--random-split` gives a stratified random split, which scores higher but is optimistic.
- Training: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau and class weights. Seed is 42.

## Project layout

```
download_data.py     download and filter RAVDESS
train.py             split, train, evaluate, save plots
app.py               Streamlit demo (upload or microphone)
src/                 config, features, augment, model
notebooks/           exploration.ipynb
assets/ models/      outputs
requirements.txt
```

## Limitations

- The clips are acted in a studio, so real speech (calls, voice notes) will do worse.
- Small dataset: about 190 clips per emotion from 24 actors. Test accuracy comes from only 4 speakers, so it moves around a lot depending on the split.
- All speakers are North American English actors saying two fixed sentences. Other accents, languages and recording conditions are not covered.
- Only 3-4 emotions, while real speech is often a mix.
- Confidence scores are not calibrated probabilities, especially for microphone audio.
- Not meant for judging people's emotional state, hiring or surveillance.

Python 3.10-3.12 recommended.
