# Quick Start Guide - Face Recognition Model

This guide will help you get started quickly with the face recognition model.

## Prerequisites

- Python 3.7 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (Deep Learning framework)
- OpenCV (Computer Vision)
- NumPy, Pandas (Data manipulation)
- Matplotlib, Seaborn (Visualization)
- Scikit-learn (ML utilities)
- Jupyter (Notebook environment)

### 2. Launch Jupyter Notebook

```bash
jupyter notebook
```

This will open Jupyter in your browser.

### 3. Open the Notebook

Navigate to and open `face_recognition_model.ipynb`

## Using the Notebook

### Step 1: Configure Your Data Path

In **Section 2**, update the data path to your UWA HSFD database location:

```python
data_path = r"C:\Users\Anvitha\Face based Person Authentication\UWA HSFD V1.1 (1)\UWA HSFD V1.1\HyperSpec_Face_Session1"
```

**Note**: If you don't have the dataset, the notebook will automatically create synthetic data for testing.

### Step 2: Run All Cells

You can run all cells sequentially:
- Click "Cell" → "Run All" in the menu, or
- Run each cell individually with `Shift + Enter`

### Step 3: Monitor Training

Watch the training progress in **Section 6**. Training typically takes:
- **With GPU**: 10-30 minutes
- **Without GPU**: 1-3 hours (depending on dataset size)

### Step 4: Evaluate Results

Check model performance in **Section 7**:
- Test accuracy
- Confusion matrix
- Per-class metrics

### Step 5: Test Authentication

Try the authentication system in **Section 8**:
- Test on random samples
- Adjust confidence threshold
- Evaluate authentication metrics

## Key Configuration Parameters

You can adjust these in the notebook:

```python
IMG_SIZE = (224, 224)  # Image dimensions
BATCH_SIZE = 32        # Training batch size
EPOCHS = 50            # Number of training epochs
```

## Model Selection

The notebook provides two model options:

### Option 1: Transfer Learning (Default - Recommended)
- Uses pre-trained ResNet50
- Faster training
- Better accuracy with less data
- **Recommended for most users**

### Option 2: Custom CNN
- Built from scratch
- More control over architecture
- Requires more data
- Longer training time

To switch models, uncomment the respective code in **Section 5**.

## Expected Results

With sufficient data, you should achieve:
- **Training Accuracy**: 95%+ 
- **Validation Accuracy**: 90%+
- **Test Accuracy**: 85-95%

Note: Results vary based on:
- Dataset size and quality
- Number of persons
- Images per person
- Training epochs

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size
```python
BATCH_SIZE = 16  # or even 8
```

### Issue: Slow Training

**Solutions**:
1. Enable GPU if available
2. Reduce image size: `IMG_SIZE = (128, 128)`
3. Reduce epochs: `EPOCHS = 30`
4. Use transfer learning (default option)

### Issue: Poor Accuracy

**Solutions**:
1. Increase training epochs
2. Add more data per person (minimum 10-20 images)
3. Ensure proper face alignment in images
4. Try different data augmentation parameters
5. Experiment with learning rate

### Issue: Dataset Not Found

**Solution**: The notebook will automatically create synthetic data for demonstration. To use real data, ensure the path is correct.

## Saving and Loading Models

The trained model is automatically saved as:
- `best_face_recognition_model.h5` (during training)
- `face_recognition_model_complete.h5` (final model)

To load a saved model:

```python
from tensorflow import keras
model = keras.models.load_model('face_recognition_model_complete.h5')
```

## Next Steps

After successful training:

1. **Test with new images**: Use the FaceAuthenticator class
2. **Adjust threshold**: Fine-tune confidence threshold for authentication
3. **Deploy**: Export model for production use
4. **Improve**: Add face detection, alignment, and anti-spoofing

## Support

For issues or questions:
1. Check the main README.md for detailed documentation
2. Review code comments in the notebook
3. Open an issue on GitHub

## Tips for Best Results

1. **Data Quality**: Ensure consistent lighting and face positioning
2. **Data Quantity**: Minimum 20 images per person recommended
3. **Balanced Dataset**: Similar number of images for each person
4. **Preprocessing**: Consider adding face detection/alignment
5. **Augmentation**: Already configured, but can be adjusted
6. **Hardware**: GPU significantly speeds up training

---

**Ready to start?** Open `face_recognition_model.ipynb` and begin!
