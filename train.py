import argparse
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src import config as C
from src.augment import spec_augment
from src.features import build_dataset
from src.model import build_model


def speaker_independent_split(actors: np.ndarray, rng: np.random.Generator):
    ids = rng.permutation(np.unique(actors))
    n_hold = max(1, round(len(ids) * 0.17))
    val_a, test_a, train_a = ids[:n_hold], ids[n_hold:2 * n_hold], ids[2 * n_hold:]
    pick = lambda group: np.where(np.isin(actors, group))[0]
    print(f"Actors  train={sorted(train_a.tolist())}  val={sorted(val_a.tolist())}  test={sorted(test_a.tolist())}")
    return pick(train_a), pick(val_a), pick(test_a)


def stratified_random_split(y: np.ndarray):
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=C.SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=C.SEED)
    return train, val, test


class SpecAugSequence(keras.utils.PyDataset):
    def __init__(self, X, y, sample_weight, batch_size, seed):
        super().__init__()
        self.X, self.y, self.w, self.bs = X, y, sample_weight, batch_size
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(X))
        self.rng.shuffle(self.order)

    def __len__(self):
        return math.ceil(len(self.X) / self.bs)

    def __getitem__(self, i):
        idx = self.order[i * self.bs:(i + 1) * self.bs]
        xb = np.stack([spec_augment(self.X[j], self.rng) for j in idx])[..., None]
        return xb, self.y[idx], self.w[idx]

    def on_epoch_end(self):
        self.rng.shuffle(self.order)


def save_curves(history, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, key, title in zip(ax, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        a.plot(history.history[key], label="train")
        a.plot(history.history["val_" + key], label="validation")
        a.set(title=title, xlabel="epoch")
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--random-split", action="store_true", help="use a stratified random split instead of by-speaker")
    args = ap.parse_args()

    C.set_seed()
    C.ASSETS_DIR.mkdir(exist_ok=True)
    C.MODELS_DIR.mkdir(exist_ok=True)

    if not (C.PROCESSED_DIR / "X.npy").exists():
        build_dataset()
    P = C.PROCESSED_DIR
    X, X_aug, y, actors = (np.load(P / f) for f in ["X.npy", "X_aug.npy", "y.npy", "actors.npy"])
    classes = json.loads((P / "labels.json").read_text())
    print(f"Data: {X.shape}, classes: {classes}")

    rng = np.random.default_rng(C.SEED)
    tr, va, te = stratified_random_split(y) if args.random_split else speaker_independent_split(actors, rng)

    X_train = np.concatenate([X[tr], X_aug[tr]])
    y_train = np.concatenate([y[tr], y[tr]])
    X_val, y_val = X[va][..., None], y[va]
    X_test, y_test = X[te][..., None], y[te]
    print(f"train={len(X_train)} (incl. augmented)  val={len(X_val)}  test={len(X_test)}")

    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_train)
    print("Class weights:", dict(zip(classes, cw.round(2))))
    train_seq = SpecAugSequence(X_train, y_train, cw[y_train].astype("float32"), args.batch_size, C.SEED)

    model = build_model(len(classes))
    ckpt = C.MODELS_DIR / "best_model.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    history = model.fit(train_seq, validation_data=(X_val, y_val), epochs=args.epochs,
                        callbacks=callbacks, verbose=2)

    best = keras.models.load_model(ckpt)
    y_pred = best.predict(X_test, verbose=0).argmax(axis=1)
    report = classification_report(y_test, y_pred, target_names=classes, digits=3)
    split_name = "random stratified" if args.random_split else "speaker-independent"
    print(f"\nTest results ({split_name} split)\n{report}")
    (C.ASSETS_DIR / "classification_report.txt").write_text(f"Split: {split_name}\n\n{report}")
    (C.MODELS_DIR / "labels.json").write_text(json.dumps(classes))

    save_curves(history, C.ASSETS_DIR / "training_curves.png")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=classes).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(C.ASSETS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Saved model -> {ckpt}\nSaved plots/report -> {C.ASSETS_DIR}")


if __name__ == "__main__":
    main()
