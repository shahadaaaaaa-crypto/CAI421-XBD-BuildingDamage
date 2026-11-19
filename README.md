CAI421 – xBD Building Damage Assessment

Project Overview

This project focuses on four-class building damage classification using paired pre-disaster and post-disaster satellite images from the xBD dataset.

The project implements and compares two deep learning models:

* Baseline ResNet-50: Uses post-disaster satellite images only to classify building damage.
* Siamese ResNet-50: Uses paired pre-disaster and post-disaster images through two branches sharing the same ResNet-50 backbone.

The main objective is to evaluate the classification performance of both models under clean image conditions and assess their robustness against different environmental image degradations.

Damage Classes

The models classify buildings into four damage categories:

1. No Damage
2. Minor Damage
3. Major Damage
4. Destroyed

Dataset

The project uses the xBD (Building Damage Assessment) satellite imagery dataset, distributed through the xView2 challenge.

A total of 600 matched pre-disaster and post-disaster image pairs were selected from the available dataset.

Building crops were extracted using the corresponding building annotations, resulting in:

* Selected image pairs: 600
* Contributing image pairs: 473
* Total building samples: 4,089
* Maximum building samples per pair: 10

Dataset Split

The dataset was divided into training, validation, and testing sets using StratifiedGroupKFold, with image pairs grouped by pair_id.

Dataset Split	Number of Samples
Training	2,756
Validation	626
Testing	707
Total	4,089

The grouping strategy ensures that samples belonging to the same image pair do not appear in different splits.

A pair-level leakage verification was performed, confirming that no image pairs overlap between the training, validation, and testing sets.

Class Distribution

Damage Class	Number of Samples
No Damage	2,524
Minor Damage	444
Major Damage	488
Destroyed	633
Total	4,089

The original xBD dataset is not included in this repository. Users must obtain the dataset separately and comply with its applicable license and usage conditions.

Data Preprocessing

The preprocessing pipeline includes:

1. Extracting pre-disaster and post-disaster RGB building crops.
2. Selecting building regions using polygon annotations.
3. Saving image crops at 512 × 512 pixels.
4. Resizing images to 224 × 224 pixels.
5. Converting images into tensors.
6. Applying ImageNet mean and standard deviation normalization.

The same preprocessing pipeline is applied to both image inputs.

Environmental Degradation

Four environmental degradation methods are implemented to evaluate model robustness:

1. Gaussian Noise
2. Atmospheric Blur
3. Resolution Reduction
4. JPEG Compression Artifacts

The degradation transformations are applied separately to both pre-disaster and post-disaster images.

A total of 32,712 degraded images are generated:

4,089 samples × 2 image views × 4 degradation types.

Important: Degraded images are generated for the dataset, but robustness evaluation uses the degraded test samples only. Both models are trained using clean images. Clean test images are also evaluated separately.

The degradation transformations are implemented using Albumentations (version 2.0.8 in the recorded experiment).

Model Architecture

1. Baseline ResNet-50

The baseline model uses only the post-disaster image as input.

The architecture consists of:

* ImageNet-pretrained ResNet-50 backbone.
* Dropout layer (0.3).
* Fully connected classification layer (2048 → 4).
* Softmax-based four-class prediction.

2. Siamese ResNet-50

The Siamese model receives paired pre-disaster and post-disaster images through two branches sharing the same ResNet-50 weights.

The extracted feature vectors are combined using:

* Absolute Difference.
* Element-wise Multiplication.
* Feature Concatenation.

The fused feature vector has 4,096 dimensions and is passed to an MLP classification head:

* Linear (4096 → 512).
* ReLU activation.
* Dropout (0.4).
* Linear (512 → 4).

The final output represents the four building damage classes.

Training Configuration

Both models are trained using the following configuration:

Parameter	Value
Optimizer	Adam
Learning Rate	0.0001
Batch Size	16
Maximum Epochs	10
Early Stopping Patience	3
Random Seed	42
Loss Function	Class-weighted Cross Entropy
Input Image Size	224 × 224
Backbone	ImageNet-pretrained ResNet-50

Early stopping monitors validation macro F1 to select the best model checkpoint.

Class weights are calculated using the training data to address class imbalance.

Evaluation

Both models are evaluated under clean and degraded test image conditions.

The evaluation metrics include:

* Accuracy
* Macro F1-score
* Weighted F1-score
* Per-class Precision
* Per-class Recall
* Per-class F1-score
* Confusion Matrix

The robustness evaluation compares model performance across the four degradation conditions.

The notebook also generates training history plots and evaluation outputs for model comparison.

Repository Contents

The repository contains the following files:

* CAI421_xBD.ipynb: Complete project implementation, including dataset preparation, preprocessing, degradation generation, model architecture, training, and evaluation.
* README.md: Project description, methodology, configuration, and execution instructions.
* requirements.txt: Required Python libraries.

Requirements

The project uses Python and the following libraries:

* NumPy
* Pandas
* Pillow
* Shapely
* Scikit-learn
* Matplotlib
* Albumentations
* PyTorch
* Torchvision
* IPython

Install the required libraries using:

pip install -r requirements.txt

How to Run the Project

The notebook is designed to run in Google Colab.

Follow these steps:

1. Open CAI421_xBD.ipynb using Google Colab.
2. Connect your Google Drive.
3. Obtain the xBD dataset and upload the required archive to your Google Drive.
4. Update the ARCHIVE_PATH variable in the first configuration cell to match the dataset archive location.
5. Enable a GPU runtime.
6. Run all notebook cells in order.

The notebook performs the following operations:

* Dataset extraction and preparation.
* Building crop generation.
* Dataset splitting and leakage verification.
* Image preprocessing.
* Environmental degradation generation.
* Baseline and Siamese model implementation.
* Model training.
* Clean test evaluation.
* Robustness evaluation.
* Results visualization and saving.

Reproducibility

A fixed random seed of 42 is used for the main Python, NumPy, and PyTorch randomization settings.

The experiment uses a batch size of 16 and an input image size of 224 × 224 pixels.

The notebook records the training configuration, validation performance, test metrics, and robustness evaluation results.

The recorded experiment was executed using a Google Colab GPU runtime.


