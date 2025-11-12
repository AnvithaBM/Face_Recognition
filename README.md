# Face Recognition using Deep Learning

A comprehensive deep learning-based face recognition system designed for the UWA Hyperspectral Face Database (UWA HSFD). This project implements a complete pipeline for face recognition and authentication using state-of-the-art deep learning techniques.

## Features

- **Deep Learning Model**: Transfer learning with ResNet50 and custom CNN architecture
- **Data Augmentation**: Advanced augmentation techniques for better generalization
- **Face Authentication**: Confidence-based authentication system with verification
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score
- **Visualization**: Training progress, confusion matrix, and result visualization
- **Model Persistence**: Save and load models for deployment
- **Production Ready**: Structured code ready for further development

## Project Structure

```
Face_Recognition/
├── face_recognition_model.ipynb  # Main Jupyter notebook with complete pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/AnvithaBM/Face_Recognition.git
cd Face_Recognition
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Open Jupyter Notebook

```bash
jupyter notebook face_recognition_model.ipynb
```

### 2. Configure Data Path

Update the data path in the notebook to point to your UWA HSFD database:

```python
data_path = r"C:\Users\Anvitha\Face based Person Authentication\UWA HSFD V1.1 (1)\UWA HSFD V1.1\HyperSpec_Face_Session1"
```

### 3. Run the Notebook

Execute cells sequentially:
- **Section 1**: Install and import dependencies
- **Section 2**: Load and preprocess hyperspectral face images
- **Section 3**: Exploratory data analysis and visualization
- **Section 4**: Data augmentation setup
- **Section 5**: Build deep learning model (Transfer Learning or Custom CNN)
- **Section 6**: Train the model with callbacks
- **Section 7**: Evaluate model performance
- **Section 8**: Face authentication and verification
- **Section 9**: Save model for deployment

## Model Architecture

### Transfer Learning Approach (Default)
- **Base Model**: ResNet50 pre-trained on ImageNet
- **Custom Layers**:
  - Global Average Pooling
  - Dense (512 units) + BatchNorm + Dropout
  - Dense (256 units) + BatchNorm + Dropout
  - Output Dense (softmax activation)

### Custom CNN Approach (Alternative)
- 4 Convolutional blocks with increasing filters (32, 64, 128, 256)
- Batch Normalization and Dropout for regularization
- Global pooling and fully connected layers
- Softmax output for classification

## Training Features

- **Data Augmentation**: Rotation, shift, zoom, flip, shear
- **Callbacks**:
  - ModelCheckpoint: Save best model
  - EarlyStopping: Prevent overfitting
  - ReduceLROnPlateau: Adaptive learning rate
- **Optimization**: Adam optimizer with learning rate scheduling
- **Fine-tuning**: Optional fine-tuning for improved performance

## Face Authentication System

The `FaceAuthenticator` class provides:

```python
# Initialize authenticator
authenticator = FaceAuthenticator(model, label_names, threshold=0.7)

# Authenticate a face
result = authenticator.authenticate('path/to/image.jpg')

# Verification mode (1:1 matching)
result = authenticator.authenticate(image, claimed_identity=person_id)
```

**Authentication Result**:
- `predicted_identity`: Recognized person name
- `confidence`: Model confidence score (0-1)
- `authenticated`: Whether confidence exceeds threshold
- `verified`: For verification mode (identity match + authenticated)

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Per-class performance visualization
- **ROC Curve**: Receiver Operating Characteristic (optional)

## Model Deployment

Trained models are saved in two formats:

1. **HDF5 Format** (`.h5`): Compatible with Keras
2. **SavedModel Format**: TensorFlow's standard format for deployment

```python
# Load saved model
from tensorflow import keras
model = keras.models.load_model('face_recognition_model_complete.h5')
```

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- OpenCV 4.6+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

See `requirements.txt` for complete list.

## Dataset

This project is designed for the **UWA Hyperspectral Face Database (UWA HSFD)**. The dataset should be organized as:

```
UWA_HSFD/
├── Person_1/
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
├── Person_2/
│   ├── image_1.png
│   └── ...
└── ...
```

If the dataset is not available, the notebook will generate synthetic data for demonstration.

## Performance Optimization

**Tips for better performance**:
1. Use GPU acceleration (CUDA-enabled GPU recommended)
2. Increase dataset size (more images per person)
3. Use data augmentation to prevent overfitting
4. Fine-tune hyperparameters (learning rate, batch size, epochs)
5. Try different architectures (VGG16, EfficientNet, etc.)
6. Implement face detection and alignment preprocessing
7. Use ensemble methods for improved accuracy

## Next Steps for Production

1. **Face Detection**: Add MTCNN or Haar Cascade for face detection
2. **Face Alignment**: Implement facial landmark detection and alignment
3. **Liveness Detection**: Add anti-spoofing measures
4. **API Development**: Create REST API using Flask/FastAPI
5. **Real-time Processing**: Optimize for video stream processing
6. **Database Integration**: Connect to user authentication database
7. **Monitoring**: Add logging and performance monitoring
8. **Security**: Implement encryption for model and data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- UWA Hyperspectral Face Database creators
- TensorFlow and Keras teams
- OpenCV community

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is a research and development project. For production deployment, additional security measures and optimization are recommended.