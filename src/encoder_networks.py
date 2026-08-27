from __future__ import annotations

import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_autoencoder(
    input_dim: int,
    latent_dim: int,
    hidden_dim: int = 64,
) -> tuple[Model, Model]:
    """
    Reproduction assumption:
    The paper specifies separate encoder networks with the same design and
    same output dimensionality, but does not specify the encoder-training
    objective or exact layer sizes. We use an autoencoder objective to learn
    each encoder independently. This is explicitly an implementation choice.
    """
    inputs = Input(shape=(input_dim,), name="input")
    hidden = Dense(hidden_dim, activation="relu", name="encoder_hidden")(inputs)
    encoded = Dense(latent_dim, activation="relu", name="encoded")(hidden)

    decoder_hidden = Dense(hidden_dim, activation="relu", name="decoder_hidden")(encoded)
    reconstructed = Dense(input_dim, activation="linear", name="reconstruction")(decoder_hidden)

    autoencoder = Model(inputs, reconstructed, name="autoencoder")
    encoder = Model(inputs, encoded, name="encoder")

    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def fit_encoder(
    X: pd.DataFrame,
    latent_dim: int,
    seed: int,
    epochs: int,
    batch_size: int = 32,
) -> tuple[np.ndarray, Model]:
    if latent_dim >= X.shape[1]:
        raise ValueError(
            f"latent_dim={latent_dim} must be smaller than input dimension "
            f"{X.shape[1]}."
        )

    set_seed(seed)
    autoencoder, encoder = build_autoencoder(
        input_dim=X.shape[1],
        latent_dim=latent_dim,
    )

    monitor = "val_loss" if len(X) >= 50 else "loss"
    validation_split = 0.1 if len(X) >= 50 else 0.0

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=10,
        restore_best_weights=True,
    )

    autoencoder.fit(
        X.values,
        X.values,
        epochs=epochs,
        batch_size=min(batch_size, len(X)),
        shuffle=True,
        verbose=0,
        validation_split=validation_split,
        callbacks=[early_stopping],
    )

    encoded = encoder.predict(X.values, verbose=0)
    return encoded, encoder


def encode_pair(
    X_source: pd.DataFrame,
    X_target: pd.DataFrame,
    latent_dim: int,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    H_source, _ = fit_encoder(
        X_source,
        latent_dim=latent_dim,
        seed=seed,
        epochs=epochs,
    )
    H_target, _ = fit_encoder(
        X_target,
        latent_dim=latent_dim,
        seed=seed + 100_000,
        epochs=epochs,
    )

    if H_source.shape[1] != H_target.shape[1]:
        raise RuntimeError("Source and target encoded dimensions do not match.")

    return H_source, H_target
