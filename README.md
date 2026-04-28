# 🚦 Traffic Sign Image Classification (CNN)

### 🔍 Overview
A **Convolutional Neural Network (CNN)** that recognizes **10 categories of traffic signs** from images. Built with **TensorFlow / Keras** and packaged for three different ways of running it: a **Colab notebook**, a **Jupyter notebook**, and a **local Python script**.

> 🎓 Originally built as part of the **CS42** *Image Classification Training* unit. Designed to teach the full ML pipeline — from raw image folders all the way to an interactive GUI predictor.

---

### 🧠 Classes
| # | Class |
|---|-------|
| 0 | Left |
| 1 | Pedestrian |
| 2 | Right |
| 3 | Roundabout |
| 4 | Speed 100 |
| 5 | Speed 120 |
| 6 | Speed 60 |
| 7 | Speed 80 |
| 8 | Stop |
| 9 | Traffic light |

---

### 📦 Features
- **End-to-end pipeline**: load → preprocess → train → evaluate → predict
- **Three execution modes** in the local script:
  1. `train` — train the CNN
  2. `predict` — classify a single image from CLI
  3. `gui` — launch a Tkinter window for interactive classification
- **Best-model checkpointing** via `val_accuracy`
- **Training accuracy plot** saved to disk
- **Pre-trained model** included (`model2.h5`) so you can predict without retraining
- Also includes a **Colab notebook** and a **Jupyter notebook** for cloud / interactive workflows

---

### 🏗️ Model Architecture
```
Input (im_size × im_size × 3)
   ↓
Conv2D(128, 3×3, ReLU)  → MaxPool(3×3)
   ↓
Conv2D(64,  3×3, ReLU)  → MaxPool(3×3)
   ↓
Flatten
   ↓
Dense(200, ReLU)
Dense(200, ReLU)
Dense(100, ReLU)
Dense( 50, ReLU)
   ↓
Dense(N_CLASSES, Softmax)
```

Optimizer: **Adam** · Loss: **sparse_categorical_crossentropy**

---

### 📁 Project Structure
```
CNN_Image_CLassification/
├── Image_classification_Traffic_Python_Local_v2.py   # Main local script (CLI subcommands)
├── Image Classification_Traffic_Colab_v3.ipynb       # Colab notebook
├── Image_Classification_Traffic_Notebook_v2.ipynb    # Jupyter notebook
├── Data/
│   ├── Training/<class_name>/*.jpg                   # Training images, organized by class folder
│   └── Validation/<class_name>/*.jpg                 # Validation images
├── CIFAR/                                            # Bonus: CIFAR-10 notebooks (separate task)
├── Database+/                                        # Larger dataset variants
├── model2.h5                                         # Pre-trained model (~4MB)
├── requirements.txt                                  # Minimum-version deps (recommended)
├── Requirements_v2.txt                               # Exact-pin deps (TF 2.12 / Py 3.10)
└── README.md
```

---

### 🧰 Requirements

**Recommended (minimum versions):**
```bash
pip install -r requirements.txt
```

**Or, the original pinned set tested with TensorFlow 2.12 / Python 3.10:**
```bash
pip install -r Requirements_v2.txt
```

The two files differ only in strictness: `requirements.txt` uses `>=` constraints (works with newer Python), `Requirements_v2.txt` uses `==` pins for an exact reproduction of the original environment.

---

### ⚙️ How to Run

#### 1) Train the model
```bash
python Image_classification_Traffic_Python_Local_v2.py train
```

With custom hyperparameters:
```bash
python Image_classification_Traffic_Python_Local_v2.py train \
  --data-dir Data \
  --im-size 64 \
  --epochs 30 \
  --batch-size 32 \
  --model-path my_model.h5 \
  --plot-path accuracy_curve.png
```

#### 2) Predict a single image
```bash
python Image_classification_Traffic_Python_Local_v2.py predict \
  --image "Data/Validation/Stop/Stop_1.jpg" \
  --model-path model2.h5
```

#### 3) Launch the GUI
```bash
python Image_classification_Traffic_Python_Local_v2.py gui --model-path model2.h5
```

---

### 🛠️ Command-Line Options

**Common to all subcommands:**
| Flag | Default | Description |
|------|---------|-------------|
| `--im-size` | `50` | Side length the images are resized to |
| `--model-path` | `model3.h5` | Where the model is saved / loaded |

**`train`:**
| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `Data` | Root folder containing `Training/` and `Validation/` subfolders |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `32` | Mini-batch size |
| `--plot-path` | `training_plot.png` | Where to save the accuracy curve PNG |

**`predict` / `gui`:**
| Flag | Default | Description |
|------|---------|-------------|
| `--image` | _(required for predict)_ | Path to the image to classify |
| `--labels` | _built-in 10 classes_ | Comma-separated class labels override |

---

### 📓 Notebook Workflows

**Colab (recommended for fast iteration):**
1. Upload the entire `CNN_Image_CLassification/` folder to **Google Drive**
2. Open `Image Classification_Traffic_Colab_v3.ipynb` in Colab
3. Mount your Drive and run the cells in order
4. Or use this shared link: <https://drive.google.com/drive/folders/1SA5AU2jgUseM41LX795Lf-a4fVtvmVgy?usp=sharing>

**Jupyter (local):**
- Open `Image_Classification_Traffic_Notebook_v2.ipynb` after `pip install -r Requirements_v2.txt`

---

### 🎓 Learning Tasks
After your first successful run, try improving the model:
1. **Augment** the data (rotations, flips, brightness)
2. Add **Dropout** to the dense layers and observe the effect on val/train gap
3. Replace the head with a **Global Average Pooling** layer
4. Try a **transfer-learning** backbone (MobileNetV2, EfficientNetB0)
5. Try a different optimizer (`SGD`, `RMSprop`) and learning rate schedule

---

### 🛑 Troubleshooting
| Issue | Fix |
|-------|-----|
| `ImportError: cannot import name 'np_utils'` | Newer Keras moved this — the refactored script uses `sparse_categorical_crossentropy` instead, no `np_utils` needed |
| GUI window doesn't open | Make sure you have a display (Tkinter requires a graphical session); not supported in pure SSH |
| Out-of-memory while training | Lower `--im-size` (e.g., 32) or `--batch-size` |
| Model file missing | Run the `train` mode first, or pass `--model-path model2.h5` to use the included pre-trained weights |

---

### 📄 License & Credits
Educational use — Credits to **[CS42.org](https://cs42.org)**.
