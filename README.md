# CAI421 – xBD Building Damage 

## Project Overview

This project focuses on classifying building damage using satellite images from the xBD dataset.

The project compares two deep learning models:

- **Baseline ResNet-50:** Uses post-disaster satellite images to classify building damage.
- **Siamese ResNet-50:** Uses paired pre-disaster and post-disaster images with a shared ResNet-50 backbone to classify building damage.

The main objective is to evaluate model performance under clean image conditions and different environmental degradation conditions.

## Damage Classes

The models classify buildings into four damage categories:

1. No Damage
2. Minor Damage
3. Major Damage
4. Destroyed

## Dataset

The project uses the xBD satellite imagery dataset.

The notebook selects 600 image pairs and extracts 4,089 building samples.

The dataset is divided into:

| Dataset Split | Number of Samples |
|---|---:|
| Training | 2,756 |
| Validation | 626 |
| Testing | 707 |
| **Total** | **4,089** |

A group-aware splitting strategy is used to prevent image-pair leakage between the training, validation, and testing sets.

The original dataset is not included in this repository. Users must obtain the xBD dataset separately and follow its license and usage conditions.

## Data Preprocessing

The preprocessing pipeline includes:

- Extracting pre-disaster and post-disaster building image crops.
- Assigning one of four damage labels to each building.
- Resizing images to 224 × 224 pixels.
- Normalizing images using ImageNet mean and standard deviation.
- Splitting the dataset into training, validation, and testing sets.

## Environmental Degradation

Four image degradation methods are applied to evaluate model robustness:

1. Gaussian Noise
2. Atmospheric Blur
3. Resolution Reduction
4. Compression Artifacts

The degradation methods are applied to both pre-disaster and post-disaster images.

A total of 32,712 degraded images are generated.

## Model Architecture

### Baseline ResNet-50

The baseline model uses a single post-disaster image as input. ResNet-50 extracts image features, which are passed to a classification layer to predict one of the four damage classes.

### Siamese ResNet-50

The Siamese model receives paired pre-disaster and post-disaster images through two branches that share the same ResNet-50 weights.

The extracted features are combined using absolute difference and element-wise multiplication before the final classification layer.

## Evaluation

The project evaluates both models under clean and degraded image conditions.

The evaluation focuses on classification performance and robustness using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Repository Contents

- `CAI421_xBD.ipynb`: Complete project implementation, including data preparation, preprocessing, degradation generation, model training, and evaluation.
- `README.md`: Project description and instructions.
- `requirements.txt`: Required Python libraries.

## Requirements

The project uses Python and the following libraries:

- NumPy
- Pandas
- Pillow
- Shapely
- Scikit-learn
- Matplotlib
- Albumentations
- PyTorch
- Torchvision
- IPython

## How to Run the Project

The notebook is designed to run in Google Colab.

1. Open `CAI421_xBD.ipynb` using Google Colab.
2. Connect your Google Drive.
3. Obtain the xBD dataset and upload the archive to your Google Drive.
4. Update the `ARCHIVE_PATH` variable in the first configuration cell to match the location of your dataset archive.
5. Make sure a GPU runtime is enabled.
6. Run the notebook cells in order.

The notebook processes the dataset, creates the required image crops and data splits, generates degraded images, trains both models, and evaluates their performance.

## Reproducibility

A fixed random seed of 42 is used for reproducibility.

The notebook also uses a batch size of 16 and an image size of 224 × 224 pixels.

## Limitations

- The experiments use a selected subset of the xBD dataset.
- Environmental degradations are synthetically generated.
- Results may vary depending on the hardware and software environment.

## Authors

CAI421 Project Team
