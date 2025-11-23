"""Training script for traffic sign image classification.

This module loads images from the ``Data/Training`` and ``Data/Validation``
directories, builds a simple convolutional neural network, and trains it
while saving the best-performing weights. The code was originally written in a
Jupyter notebook; it has been refactored into reusable functions with clearer
comments to make it easier to run locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import set_random_seed

# Constants ---------------------------------------------------------------------
IM_SIZE: int = 50
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Data"
TRAIN_DIR = DATA_ROOT / "Training"
VAL_DIR = DATA_ROOT / "Validation"
MODEL_PATH = PROJECT_ROOT / "model3.h5"
RANDOM_SEED = 42


# Utility functions -------------------------------------------------------------
def get_class_names(directory: Path) -> List[str]:
    """Return a sorted list of class folder names found in ``directory``.

    Args:
        directory: Path containing one subdirectory per class.

    Raises:
        FileNotFoundError: If the provided directory does not exist.
    """

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted([entry.name for entry in directory.iterdir() if entry.is_dir()])


def process_image(image_path: Path, image_size: int) -> np.ndarray:
    """Load, resize, and normalize a single image.

    Args:
        image_path: Path to the image file.
        image_size: Target size to resize the image to (square output).

    Returns:
        A NumPy array with shape ``(image_size, image_size, 3)`` and pixel values
        scaled to ``[0, 1]``.
    """

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    resized = cv2.resize(image, (image_size, image_size))
    normalized = resized.astype("float32") / 255.0
    return normalized


def load_dataset(directory: Path, class_names: Iterable[str], image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load all images from a dataset directory.

    The function expects a ``directory`` structure containing one folder per
    class. Each folder should contain the images for that class.

    Args:
        directory: Root directory for the dataset split (training or validation).
        class_names: Iterable of class folder names to load.
        image_size: Target image size for resizing.

    Returns:
        Tuple of ``(images, labels)`` where ``images`` has shape
        ``(num_images, image_size, image_size, 3)`` and ``labels`` has shape
        ``(num_images,)``.
    """

    images: List[np.ndarray] = []
    labels: List[int] = []

    for class_index, class_name in enumerate(class_names):
        class_dir = directory / class_name
        if not class_dir.exists():
            print(f"Warning: class directory missing and will be skipped: {class_dir}")
            continue

        for image_file in sorted(class_dir.iterdir()):
            if not image_file.is_file():
                continue
            try:
                images.append(process_image(image_file, image_size))
                labels.append(class_index)
            except Exception as exc:  # noqa: BLE001 - log and continue on processing errors
                print(f"Skipping {image_file} due to error: {exc}")

    if not images:
        raise ValueError(f"No images were loaded from {directory}")

    X = np.stack(images).astype("float32")
    y = np.array(labels, dtype="int32")
    return X, y


def build_model(image_size: int, num_classes: int) -> Sequential:
    """Create the convolutional neural network architecture."""

    model = Sequential(
        [
            Conv2D(128, (3, 3), strides=1, activation="relu", input_shape=(image_size, image_size, 3)),
            MaxPooling2D(pool_size=(3, 3)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(3, 3)),
            Flatten(),
            Dense(200, activation="relu"),
            Dropout(0.1),
            Dense(200, activation="relu"),
            Dense(100, activation="relu"),
            Dense(50, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_history(history: tf.keras.callbacks.History) -> None:
    """Plot training and validation accuracy/loss curves."""

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "r", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "r", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def predict_image(model: tf.keras.Model, image_path: Path, class_names: List[str]) -> str:
    """Run a single-image prediction and return the class name."""

    processed = process_image(image_path, IM_SIZE)
    processed = processed.reshape(1, IM_SIZE, IM_SIZE, 3)
    prediction = model.predict(processed, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])

    print(f"Predicted class: {class_names[predicted_index]} ({confidence:.2%} confidence)")
    return class_names[predicted_index]


# Training pipeline -------------------------------------------------------------
def train() -> None:
    """Load data, train the CNN, and save the best-performing model."""

    set_random_seed(RANDOM_SEED)

    class_names = get_class_names(TRAIN_DIR)
    print(f"Found classes: {class_names}")

    print("\nLoading training data...")
    X_train, y_train = load_dataset(TRAIN_DIR, class_names, IM_SIZE)
    print(f"Loaded {len(y_train)} training images with shape {X_train.shape}")

    print("\nLoading validation data...")
    X_val, y_val = load_dataset(VAL_DIR, class_names, IM_SIZE)
    print(f"Loaded {len(y_val)} validation images with shape {X_val.shape}")

    model = build_model(IM_SIZE, len(class_names))
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],
        verbose=1,
    )

    plot_history(history)

    # Optionally test a single prediction to verify the saved model.
    sample_class_dir = VAL_DIR / class_names[0]
    sample_image = next(iter(sample_class_dir.glob("*")), None)
    if sample_image:
        best_model = tf.keras.models.load_model(MODEL_PATH)
        predict_image(best_model, sample_image, class_names)


if __name__ == "__main__":
    train()
