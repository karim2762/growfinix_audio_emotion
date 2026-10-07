from tensorflow import keras
from tensorflow.keras import layers

from . import config as C


def conv_block(x, filters: int, dropout: float):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    return layers.Dropout(dropout)(x)


def build_model(n_classes: int, input_shape=(C.N_MELS, C.N_FRAMES, 1)) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = conv_block(inputs, 16, 0.25)
    x = conv_block(x, 32, 0.25)
    x = conv_block(x, 64, 0.30)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="emotion_cnn")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


if __name__ == "__main__":
    build_model(3).summary()
