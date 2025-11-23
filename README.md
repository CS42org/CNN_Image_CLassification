# Traffic Sign CNN Classification

This repository contains a TensorFlow/Keras training script for traffic sign image classification. The code loads images from local directories, trains a convolutional neural network, and saves the best-performing model checkpoint.

## Project structure
- `Image_classification_Traffic_Python_Local_v2.py`: Refactored training script with reusable functions for loading data, building the CNN, training, and running single-image predictions.
- `Data/Training/`: One subfolder per class containing training images.
- `Data/Validation/`: One subfolder per class containing validation images (same class names as training).
- `model3.h5`: Saved model checkpoint produced by the training script (created after training completes).
- `Requirements_v2.txt`: Python dependency list.

## Environment setup
1. **Python version**: Tested with Python 3.10+.
2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r Requirements_v2.txt
   ```

## Prepare the dataset
Ensure your data is placed in the following structure:
```
Data/
  Training/
    class_a/
      image1.jpg
      ...
    class_b/
      ...
  Validation/
    class_a/
      ...
    class_b/
      ...
```
- Folder names under `Training` define the class labels and must match the folders under `Validation`.
- Images are resized to 50x50 pixels during preprocessing.

## Run training locally
From the repository root, run:
```bash
python Image_classification_Traffic_Python_Local_v2.py
```
The script will:
1. Discover class names from `Data/Training`.
2. Load and normalize images from training and validation folders.
3. Build a CNN, train for 50 epochs, and save the best checkpoint to `model3.h5` based on validation accuracy.
4. Display training/validation accuracy and loss plots at the end of training.

## Optional: quick single-image prediction
After training finishes, the script automatically loads the saved `model3.h5` file and prints a prediction for the first image found in the first validation class folder. To test another image manually, you can use the helper function inside the script:
```python
from pathlib import Path
import tensorflow as tf
from Image_classification_Traffic_Python_Local_v2 import predict_image, get_class_names, IM_SIZE, VAL_DIR, MODEL_PATH

model = tf.keras.models.load_model(MODEL_PATH)
class_names = get_class_names(VAL_DIR)
print(predict_image(model, Path("path/to/your/image.jpg"), class_names))
```

## Troubleshooting
- **Missing data folders**: Ensure `Data/Training` and `Data/Validation` exist and contain class subfolders with images.
- **Empty dataset warning**: If no images are loaded, confirm file extensions are supported and paths are correct.
- **GPU usage**: TensorFlow will use available GPUs automatically. To force CPU, set `CUDA_VISIBLE_DEVICES=""` before running the script.
