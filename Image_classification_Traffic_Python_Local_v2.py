"""
Traffic Sign Image Classification — Local Python Script
-------------------------------------------------------
Builds, trains, and runs inference for a small Convolutional Neural Network
that recognizes 10 traffic-sign classes:
    Left, Pedestrian, Right, Roundabout, Speed 100,
    Speed 120, Speed 60, Speed 80, Stop, Traffic light

Three modes:
  1) train    — train the CNN from images in --data-dir
  2) predict  — classify a single image file
  3) gui      — open a small Tkinter window to load and classify any image

Examples:
  # Train (defaults: ./Data, im_size=50, 50 epochs)
  python Image_classification_Traffic_Python_Local_v2.py train

  # Train with custom hyperparameters
  python Image_classification_Traffic_Python_Local_v2.py train --epochs 30 --im-size 64 --batch-size 32

  # Predict a single image
  python Image_classification_Traffic_Python_Local_v2.py predict --image "Data/Validation/Stop/Stop_1.jpg"

  # Launch the Tkinter GUI
  python Image_classification_Traffic_Python_Local_v2.py gui
"""

import argparse
import os
import random
import sys
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

# Default class labels — used if the model is loaded without retraining
DEFAULT_LABELS = [
    "Left", "Pedestrian", "Right", "Roundabout",
    "Speed 100", "Speed 120", "Speed 60", "Speed 80",
    "Stop", "Traffic light",
]


# ----------------------------- Data loading -----------------------------

def list_class_folders(path: str) -> List[str]:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset folder does not exist: {path}")
    return sorted([d for d in next(os.walk(path))[1]])


def preprocess_image(file_path: str, im_size: int) -> np.ndarray:
    im = cv2.imread(file_path)
    if im is None:
        raise ValueError(f"Could not read image: {file_path}")
    im = cv2.resize(im, (im_size, im_size))
    im = im / 255.0
    return im


def load_split(split_dir: str, classes: List[str], im_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[!] Missing class folder, skipping: {class_dir}")
            continue
        for fname in next(os.walk(class_dir))[2]:
            full_path = os.path.join(class_dir, fname)
            try:
                X.append(preprocess_image(full_path, im_size))
                y.append(idx)
            except Exception as e:
                print(f"[!] Skipping {full_path}: {e}")

    if not X:
        raise RuntimeError(f"No images loaded from {split_dir}")

    X = np.asarray(X, dtype=np.float32).reshape(-1, im_size, im_size, 3)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y


# ----------------------------- Model -----------------------------

def build_model(im_size: int, n_classes: int):
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Conv2D(128, (3, 3), strides=1, input_shape=(im_size, im_size, 3), activation="relu"),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(3, 3)),
        Flatten(),
        Dense(200, activation="relu"),
        Dense(200, activation="relu"),
        Dense(100, activation="relu"),
        Dense(50, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ----------------------------- Mode: train -----------------------------

def cmd_train(args):
    import tensorflow as tf

    train_dir = os.path.join(args.data_dir, "Training")
    val_dir = os.path.join(args.data_dir, "Validation")

    classes = list_class_folders(train_dir)
    print(f"Found {len(classes)} classes: {classes}")

    print("Loading training data...")
    X_train, y_train = load_split(train_dir, classes, args.im_size)
    print(f"Training shape: {X_train.shape}")

    print("Loading validation data...")
    X_test, y_test = load_split(val_dir, classes, args.im_size)
    print(f"Validation shape: {X_test.shape}")

    # Shuffle the training set
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    model = build_model(args.im_size, len(classes))
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.model_path, monitor="val_accuracy", save_best_only=True
    )
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint],
        verbose=1,
    )

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final validation accuracy: {acc * 100:.2f}%")
    print(f"Best model saved to: {args.model_path}")

    # Save training plot
    if args.plot_path:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[!] matplotlib not installed; skipping training plot.")
            return
        epochs = range(len(history.history["accuracy"]))
        plt.plot(epochs, history.history["accuracy"], "r", label="Training accuracy")
        plt.plot(epochs, history.history["val_accuracy"], "b", label="Validation accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend(loc=0)
        plt.savefig(args.plot_path, dpi=150, bbox_inches="tight")
        print(f"Training plot saved to: {args.plot_path}")


# ----------------------------- Mode: predict -----------------------------

def predict_one(model, im_array: np.ndarray, im_size: int, labels: List[str]) -> str:
    im = im_array.reshape(1, im_size, im_size, 3)
    pred = model.predict(im, verbose=0).reshape(-1)
    idx = int(np.argmax(pred))
    class_name = labels[idx]

    pct = (pred / pred.sum() * 100).astype("int32")
    table = pd.DataFrame([pct], columns=labels)
    print("Predicted class:", class_name)
    print("Class probabilities (%):")
    print(table.to_string(index=False))
    return class_name


def cmd_predict(args):
    from tensorflow.keras.models import load_model

    if not os.path.exists(args.model_path):
        print(f"[!] Model not found at {args.model_path}. Train first or pass --model-path.", file=sys.stderr)
        sys.exit(1)

    model = load_model(args.model_path)
    labels = DEFAULT_LABELS if args.labels is None else args.labels.split(",")
    im = preprocess_image(args.image, args.im_size)
    predict_one(model, im, args.im_size, labels)


# ----------------------------- Mode: gui -----------------------------

def cmd_gui(args):
    import tkinter as tk
    from tkinter import Button, Label, filedialog

    from PIL import Image, ImageTk
    from tensorflow.keras.models import load_model

    if not os.path.exists(args.model_path):
        print(f"[!] Model not found at {args.model_path}. Train first.", file=sys.stderr)
        sys.exit(1)

    model = load_model(args.model_path)
    labels = DEFAULT_LABELS if args.labels is None else args.labels.split(",")

    top = tk.Tk()
    top.geometry("800x600")
    top.title("Traffic Sign Image Classification")
    top.configure(background="#CDCDCD")

    heading = Label(top, text="Image Classification", pady=15, font=("arial", 20, "bold"))
    heading.configure(background="#CDCDCD", foreground="#364156")
    heading.pack()

    label_widget = Label(top, background="#CDCDCD", font=("arial", 15, "bold"))
    sign_image = Label(top)

    def classify(file_path):
        im = preprocess_image(file_path, args.im_size)
        cls = predict_one(model, im, args.im_size, labels)
        label_widget.configure(foreground="#011638", text=cls)

    def show_classify_button(file_path):
        btn = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
        btn.configure(background="#364156", foreground="white", font=("arial", 10, "bold"))
        btn.place(relx=0.79, rely=0.46)

    def upload_image():
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
            if not file_path:
                return
            uploaded = Image.open(file_path)
            uploaded.thumbnail((top.winfo_width() / 2.25, top.winfo_height() / 2.25))
            im_tk = ImageTk.PhotoImage(uploaded)
            sign_image.configure(image=im_tk)
            sign_image.image = im_tk
            label_widget.configure(text="")
            show_classify_button(file_path)
        except Exception as e:
            print(f"[!] {e}", file=sys.stderr)

    upload = Button(top, text="Choose an image", command=upload_image, padx=10, pady=5)
    upload.configure(background="#364156", foreground="white", font=("arial", 10, "bold"))
    upload.pack(side="bottom", pady=50)
    sign_image.pack(side="bottom", expand=True)
    label_widget.pack(side="bottom", expand=True)

    top.mainloop()


# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Traffic Sign CNN — train / predict / gui")
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--im-size", type=int, default=50, help="Image side length (default: 50)")
    common.add_argument("--model-path", default="model3.h5", help="Where to save/load the model (default: model3.h5)")

    pt = sub.add_parser("train", parents=[common], help="Train the CNN")
    pt.add_argument("--data-dir", default="Data", help="Folder with Training/ and Validation/ subfolders (default: Data)")
    pt.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    pt.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    pt.add_argument("--plot-path", default="training_plot.png", help="Where to save the training-curve plot (default: training_plot.png)")
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", parents=[common], help="Classify a single image")
    pp.add_argument("--image", required=True, help="Path to the image file")
    pp.add_argument("--labels", default=None, help="Comma-separated class labels (overrides defaults)")
    pp.set_defaults(func=cmd_predict)

    pg = sub.add_parser("gui", parents=[common], help="Open the Tkinter GUI")
    pg.add_argument("--labels", default=None, help="Comma-separated class labels (overrides defaults)")
    pg.set_defaults(func=cmd_gui)

    return p.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
