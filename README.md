# Face Recognition with Hyperspectral Data

This repository contains a comprehensive Jupyter notebook implementing a deep learning-based face recognition system using hyperspectral face data from the UWA HSFD dataset.

## Overview

The project builds a CNN-based face recognition model adapted for hyperspectral imaging data with multiple spectral bands. The implementation includes:

- **Data Loading & Preprocessing**: Custom loaders for hyperspectral images handling multiple spectral bands
- **Exploratory Data Analysis**: Comprehensive analysis of dataset characteristics and distributions
- **Deep Learning Model**: CNN architecture optimized for hyperspectral face recognition
- **Training Pipeline**: Complete training setup with data augmentation and advanced callbacks
- **Evaluation Metrics**: Detailed performance analysis including accuracy, precision, recall, and F1-score
- **Visualization**: Rich visualizations of training progress, predictions, and model performance

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- TensorFlow/Keras (>=2.10.0)
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenCV, Pillow
- Scikit-learn
- Jupyter Notebook

## Dataset

The notebook is configured to work with the **UWA HSFD V1.1** (Hyperspectral Face Database) dataset. The default path is:
```
C:\Users\Anvitha\Face based Person Authentication\UWA HSFD V1.1 (1)\UWA HSFD V1.1\HyperSpec_Face_Session1
```

**Note**: If the dataset is not available at the specified path, the notebook will automatically create a synthetic dataset for demonstration purposes.

### Dataset Structure Expected:
```
HyperSpec_Face_Session1/
├── Person_01/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── Person_02/
│   ├── image1.png
│   └── ...
└── ...
```

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AnvithaBM/Face_Recognition.git
   cd Face_Recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook face_recognition_hyperspectral.ipynb
   ```

4. **Configure dataset path** (if different):
   - Open the notebook
   - Locate the "Dataset Configuration" section
   - Update the `DATASET_PATH` variable with your dataset location

5. **Run all cells** to:
   - Load and preprocess the data
   - Perform exploratory data analysis
   - Build and train the model
   - Evaluate performance
   - Visualize results

## Notebook Structure

The notebook is organized into the following sections:

1. **Setup and Imports**: Import all necessary libraries
2. **Data Loading and Preprocessing**: Custom loaders for hyperspectral images
3. **Exploratory Data Analysis**: Dataset statistics and visualizations
4. **Model Architecture**: CNN model design for hyperspectral data
5. **Training**: Model training with data augmentation
6. **Evaluation**: Comprehensive performance metrics
7. **Visualization and Results**: Detailed visualizations and analysis

## Model Architecture

The CNN model features:
- 4 convolutional blocks with batch normalization and dropout
- Progressive filter sizes (32 → 64 → 128 → 256)
- Dense layers with regularization
- Softmax output for multi-class classification

## Key Features

✅ **Hyperspectral Image Support**: Handles multi-band spectral data  
✅ **Data Augmentation**: Rotation, shifting, zooming, and flipping  
✅ **Advanced Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  
✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score  
✅ **Rich Visualizations**: Training history, confusion matrix, predictions  
✅ **Model Persistence**: Saves trained model and artifacts for deployment  
✅ **Synthetic Data Support**: Automatic fallback for demo purposes  

## Model Output

The trained model produces:
- `best_face_recognition_model.h5` - Best model during training
- `hyperspectral_face_recognition_model.h5` - Final trained model
- `label_encoder.pkl` - Label encoder for inference
- `training_history.csv` - Training metrics history

## Performance

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of actual positives identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## Customization

You can customize the model by modifying:

- **Image size**: `IMG_HEIGHT`, `IMG_WIDTH` (default: 128x128)
- **Batch size**: `BATCH_SIZE` (default: 32)
- **Training epochs**: `EPOCHS` (default: 50)
- **Learning rate**: `LEARNING_RATE` (default: 0.001)
- **Model architecture**: Modify the `build_face_recognition_model()` function

## Deployment

The trained model is suitable for:
- Face authentication systems
- Access control applications
- Biometric identification
- Security systems

To use the model in production:
```python
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model('hyperspectral_face_recognition_model.h5')

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Make predictions
predictions = model.predict(new_images)
predicted_labels = label_encoder.inverse_transform(predictions.argmax(axis=1))
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open-source and available under the MIT License.

## Author

**Anvitha BM**

## Acknowledgments

- UWA HSFD Dataset creators
- TensorFlow/Keras community
- Open-source contributors