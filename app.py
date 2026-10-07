import io
import json

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config as C
from src.features import load_audio, log_mel, preprocess_waveform, normalize

st.set_page_config(page_title="Speech Emotion Classifier", page_icon="🎙️")
st.title("🎙️ Speech Emotion Classifier")
st.caption("A CNN looks at a log-Mel spectrogram (an 'image of sound') and guesses the emotion.")


@st.cache_resource
def load_model():
    import keras
    model_path, labels_path = C.MODELS_DIR / "best_model.keras", C.MODELS_DIR / "labels.json"
    if not model_path.exists():
        return None, None
    return keras.models.load_model(model_path), json.loads(labels_path.read_text())


model, labels = load_model()
if model is None:
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

tab_upload, tab_mic = st.tabs(["Upload .wav", "Record"])
with tab_upload:
    uploaded = st.file_uploader("Choose a .wav file", type=["wav"])
with tab_mic:
    recorded = st.audio_input("Record a few seconds of speech")

audio_file = uploaded or recorded
if audio_file is None:
    st.info("Upload a .wav file or record something to begin.")
    st.stop()

st.audio(audio_file)
y = preprocess_waveform(load_audio(io.BytesIO(audio_file.getvalue())))
spec_db = log_mel(y)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Waveform")
    fig, ax = plt.subplots(figsize=(5, 3))
    librosa.display.waveshow(y, sr=C.SAMPLE_RATE, ax=ax)
    ax.set(xlabel="Time (s)", ylabel="Amplitude")
    st.pyplot(fig)
with col2:
    st.subheader("Log-Mel spectrogram")
    fig, ax = plt.subplots(figsize=(5, 3))
    img = librosa.display.specshow(spec_db, sr=C.SAMPLE_RATE, hop_length=C.HOP_LENGTH,
                                   x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+0.0f dB")
    st.pyplot(fig)

x = normalize(spec_db)[None, ..., None]
probs = model.predict(x, verbose=0)[0]
best = int(probs.argmax())
st.subheader(f"Prediction: **{labels[best].upper()}** ({probs[best]:.0%} confidence)")
st.bar_chart(pd.DataFrame({"probability": probs}, index=labels))
st.caption("⚠️ Trained on ~600 *acted* clips from 24 North-American actors. "
           "Treat results as a demo, not a diagnosis. See README → Limitations.")
