"""A small CNN that reads a spectrogram like an image.

WHY a CNN: a spectrogram is an image whose patterns (pitch contours, energy
bursts) are local and can appear anywhere in time - exactly what convolutions
are good at detecting.
"""
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C


def conv_block(x, filters: int, dropout: float):
    """Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout.

    BatchNorm: keeps activations well-scaled so training is faster and steadier.
    MaxPool:   halves the image size (cheaper) and keeps the strongest signals.
    Dropout:   randomly switches off units so the net can't memorise.
    """
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)  # bias is redundant before BatchNorm
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    return layers.Dropout(dropout)(x)


def build_model(n_classes: int, input_shape=(C.N_MELS, C.N_FRAMES, 1)) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = conv_block(inputs, 16, 0.25)
    x = conv_block(x, 32, 0.25)
    x = conv_block(x, 64, 0.30)
    # Global average pooling instead of Flatten: far fewer weights (less overfitting,
    # faster on CPU) and the model doesn't care *when* in the clip a pattern occurs.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)  # probabilities that sum to 1

    model = keras.Model(inputs, outputs, name="emotion_cnn")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",  # labels are ints 0..n-1
                  metrics=["accuracy"])
    return model


if __name__ == "__main__":
    build_model(3).summary()
