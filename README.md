# Plant Disease Classifier

A practice project: classifying plant leaf diseases from photos with a small convolutional network, built with TensorFlow/Keras on the PlantVillage dataset.

The baseline CNN reaches **94.4% validation accuracy** across 38 classes after 10 epochs of CPU training.

## Dataset

[PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) leaf images: 14 crops, 38 classes (each a crop paired with a disease or `healthy`).

| Split | Images |
| --- | ---: |
| Train | 43,444 |
| Validation | 10,861 |

Every image is 256×256 RGB. The classes are imbalanced: validation support ranges from 31 images (`Potato___healthy`) to 1,102 (`Orange___Haunglongbing_(Citrus_greening)`).

The data is not in the repository. The notebook expects one folder per class under each split:

```
data/PlantVillage/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── ...
└── val/
    ├── Apple___Apple_scab/
    └── ...
```

## Setup

Developed on Python 3.9 with TensorFlow 2.20.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt scikit-learn
python -m ipykernel install --user --name plant-disease-classifier
jupyter lab plant-disease-classifier.ipynb
```

`scikit-learn` is used for the per-class evaluation and is not yet listed in `requirements.txt`.

## What the notebook does

1. **Exploratory data analysis**: class count, images per class, image sizes, one sample image per class.
2. **Input pipeline**: `image_dataset_from_directory` at 128×128 with batch size 32, cached as `uint8`, shuffled each epoch, rescaled to 0–1, and prefetched. Caching cuts a full pass over the training set from 6.5 s to 0.9 s.
3. **Baseline CNN**: the model below, compiled with Adam and categorical cross-entropy.
4. **Sanity checks before training**:
   - The untrained loss is 3.65, matching the expected `ln(38) ≈ 3.64` for a uniform guess.
   - The model can overfit a single batch (loss 3.65 → 0.01 in 50 epochs).
   - A short run on 100 batches confirms the loss falls on real data.
5. **Full training**: 10 epochs on the full training set.
6. **Evaluation**: loss and accuracy curves, a per-class classification report, a confusion matrix, and the most frequent misclassifications.

Running it writes `baseline.keras` (the trained model) and `history_baseline.json` (per-epoch metrics).

## Model

| Layer | Output shape | Parameters |
| --- | --- | ---: |
| Input | 128×128×3 | 0 |
| Conv2D, 32 filters, 3×3, ReLU | 128×128×32 | 896 |
| MaxPooling2D, 3×3 | 42×42×32 | 0 |
| Conv2D, 64 filters, 3×3, ReLU | 42×42×64 | 18,496 |
| MaxPooling2D, 3×3 | 14×14×64 | 0 |
| Conv2D, 128 filters, 3×3, ReLU | 14×14×128 | 73,856 |
| MaxPooling2D, 3×3 | 4×4×128 | 0 |
| Flatten | 2,048 | 0 |
| Dense, 128 units, ReLU | 128 | 262,272 |
| Dense, 38 units, softmax | 38 | 4,902 |

Total: 360,422 parameters (1.37 MB).

## Results

Metrics after the final epoch (epoch 10), which is the model saved to `baseline.keras`:

| Metric | Train | Validation |
| --- | ---: | ---: |
| Accuracy | 0.974 | 0.944 |
| Loss | 0.074 | 0.192 |
| Precision | 0.976 | 0.949 |
| Recall | 0.972 | 0.943 |

On the validation set the macro-averaged F1 is 0.924 and the weighted F1 is 0.944.

Validation accuracy peaked at 0.952 in epoch 9. Training accuracy kept climbing while validation accuracy levelled off and fluctuated from epoch 6 onward, so the model is starting to overfit.

Training took about 98 seconds per epoch on CPU, roughly 16 minutes in total.

### Where it struggles

Most errors are between diseases of the same crop that look alike. The largest confusions on the validation set:

| True class | Predicted as | Count | Share of true class |
| --- | --- | ---: | ---: |
| Corn: Cercospora / gray leaf spot | Corn: northern leaf blight | 35 | 34.0% |
| Grape: esca (black measles) | Grape: black rot | 26 | 9.4% |
| Tomato: spider mites | Tomato: target spot | 21 | 6.3% |
| Tomato: late blight | Tomato: early blight | 16 | 4.2% |
| Tomato: target spot | Tomato: spider mites | 13 | 4.6% |

The weakest classes by F1 are corn gray leaf spot (0.734, recall 0.631) and tomato early blight (0.781, recall 0.715).

## Using the trained model

The model takes 128×128 RGB images scaled to 0–1. Class indices follow the alphabetical order of the folder names under `train/`.

```python
import os
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("baseline.keras")
class_names = sorted(
    d for d in os.listdir("data/PlantVillage/train")
    if os.path.isdir(os.path.join("data/PlantVillage/train", d))
)

img = tf.keras.utils.load_img("leaf.jpg", target_size=(128, 128))
x = np.expand_dims(tf.keras.utils.img_to_array(img) / 255.0, axis=0)

probs = model.predict(x)[0]
print(class_names[np.argmax(probs)], f"{probs.max():.1%}")
```

## Limitations

- PlantVillage photos show single leaves against plain backgrounds, so accuracy on photos taken in the field is likely to be lower.
- There is no held-out test set. The reported numbers come from the validation split.
- The baseline uses no data augmentation, regularisation, class weighting, or checkpointing of the best epoch.
