# Face Recognition using Hyperspectral Data

This repository implements a face recognition system using hyperspectral face data with Gabor transform feature extraction and deep learning (MobileNet/VGG16).

## Features

- **Hyperspectral Image Processing**: Load and preprocess hyperspectral face images
- **Gabor Transform**: Apply Gabor filters to extract texture features
- **Transfer Learning**: Use pre-trained MobileNet or VGG16 models for face recognition
- **Dual-Branch Architecture**: Process RGB and Gabor features separately, then combine
- **Comprehensive Evaluation**: Accuracy metrics, confusion matrix, and visualizations

## Implementation Details

The system follows the approach described in `Gabor_CNN.pdf`:

1. **Data Loading**: Loads hyperspectral face images from the dataset
2. **Gabor Transform**: Applies Gabor filters to extract texture features
3. **CNN Model**: Uses MobileNet or VGG16 with transfer learning
4. **Training**: Trains the model with data augmentation
5. **Evaluation**: Evaluates accuracy and generates performance metrics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook "face_recognition_hyperspectral (3).ipynb"
```

The notebook includes:
- Data loading and preprocessing
- Gabor transform implementation
- Model architecture (MobileNet/VGG16)
- Training pipeline
- Evaluation and visualization

## Dataset

The implementation expects hyperspectral face data in a directory structure where each subdirectory represents a person/class. If the dataset is not available, the notebook includes code to generate synthetic data for demonstration purposes.

## Model Options

The notebook supports two pre-trained models:
- **MobileNet**: Lightweight and efficient (default)
- **VGG16**: Deeper architecture, potentially better accuracy

Change the `model_type` parameter in the notebook to switch between models.

## Results

The notebook generates:
- Training/validation accuracy and loss curves
- Confusion matrix
- Classification report with precision, recall, F1-score
- Sample predictions with visualizations
- Saved model for deployment

## References

- Research paper: `Gabor_CNN.pdf`
- TensorFlow/Keras documentation
- MobileNet and VGG16 architectures