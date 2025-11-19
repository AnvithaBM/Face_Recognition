# Model Information

## Required Model File

The face authentication system requires a trained model file named `best_model.keras`.

### Model Specifications

- **Format**: Keras model (.keras format)
- **Architecture**: VGG-based CNN
- **Input Shape**: (128, 128, 3) - RGB images after Gabor transformation
- **Output**: Softmax classification layer
- **Feature Layer**: The penultimate layer (before softmax) should output 256-dimensional embeddings

### How to Obtain the Model

#### Option 1: Train Using the Provided Notebook

1. Open `face_recognition_hyperspectral (3).ipynb`
2. Follow the notebook cells to:
   - Load and preprocess your dataset
   - Apply Gabor filters
   - Build the VGG model
   - Train on your face dataset
   - Save as `best_model.keras`

#### Option 2: Model Architecture Reference

If you need to create your own model, it should follow this structure:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential([
    # VGG-like convolutional blocks
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    # Flatten and dense layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),  # This is the feature extraction layer
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  # Classification layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Important Notes

1. **Feature Extraction**: The system extracts features from the layer before the final softmax layer (layer index -3 in app.py)

2. **Pre-processing**: All input images are:
   - Resized to 128x128
   - Converted to RGB
   - Processed with Gabor filters (4 orientations)
   - Normalized to [0, 1]

3. **Training Data**: 
   - Should contain multiple images per person
   - Various lighting conditions and angles
   - Clear face images

4. **Without Model**: The application will run without the model file, but:
   - Registration will fail
   - Authentication will fail
   - You'll see a warning message in the console

### Gabor Transform Parameters

The model expects inputs transformed with these Gabor filter parameters:

```python
GABOR_PARAMS = {
    'ksize': 31,           # Kernel size
    'sigma': 4.0,          # Standard deviation
    'theta_values': [0, π/4, π/2, 3π/4],  # Orientations
    'lambda': 10.0,        # Wavelength
    'gamma': 0.5,          # Aspect ratio
    'psi': 0               # Phase offset
}
```

### Testing the Model

Once you have the model file:

1. Place it in the project root directory as `best_model.keras`
2. Restart the Flask application
3. The console should show: "Model loaded from best_model.keras"
4. Try registering a user with 5-10 face images
5. Test authentication

### Model Performance

Expected performance metrics:
- **Accuracy**: Should be >90% on validation set
- **Feature Dimension**: 256
- **Inference Time**: ~100-500ms per image (CPU)
- **Authentication Threshold**: 0.7 (70% similarity)

You can adjust the authentication threshold in `app.py` line 212:
```python
threshold = 0.7  # Adjust as needed
```

### Troubleshooting

**Issue**: Model loads but authentication fails
- Check that the model was trained with Gabor-transformed images
- Verify the feature extraction layer index is correct
- Ensure sufficient training data was used

**Issue**: Low accuracy
- Increase training data
- Use data augmentation
- Adjust Gabor parameters
- Fine-tune the model architecture

**Issue**: Model file too large
- Use model quantization
- Reduce model complexity
- Use mixed precision training
