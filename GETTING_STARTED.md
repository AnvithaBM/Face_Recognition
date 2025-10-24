# Getting Started with Hyperspectral Face Authentication

This guide will help you get started with the Hyperspectral Face Recognition and Authentication System.

## Quick Start (5 minutes)

Want to see it in action right away? Run the quick demo:

```bash
python quick_demo.py
```

This will:
1. Create a test model
2. Enroll 3 synthetic users
3. Demonstrate 1:1 verification and 1:N identification
4. Show database management operations

## Full Workflow

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

Open and run the model training notebook:

```bash
jupyter notebook hyperspectral_face_recognition_model.ipynb
```

This notebook will:
- Load or generate hyperspectral face data
- Build and train a CNN-based embedding model
- Save the trained model as `hyperspectral_embedding_model.h5`
- Generate training metrics and visualizations

**Important**: The notebook includes synthetic data for demonstration. For real use, update the `DATA_PATH` to point to your hyperspectral face dataset.

### Step 3: Use the Authentication System

#### Option A: Interactive Notebook

Explore the system interactively:

```bash
jupyter notebook face_authentication_demo.ipynb
```

This notebook demonstrates:
- System initialization
- User enrollment with multiple samples
- 1:1 verification (specific user)
- 1:N identification (find best match)
- Threshold tuning
- Database management
- Performance analysis

#### Option B: Command Line

Use the command-line interface:

```bash
# Run full demo with synthetic data
python example_usage.py --demo

# Enroll a new user
python example_usage.py --enroll alice --images face1.npy face2.npy face3.npy

# Authenticate (1:1 verification)
python example_usage.py --authenticate test.npy --user alice

# Identify (1:N identification)
python example_usage.py --authenticate test.npy

# List all users
python example_usage.py --list

# Remove a user
python example_usage.py --remove alice
```

#### Option C: Python API

Integrate into your application:

```python
from face_authentication_system import HyperspectralFaceAuthenticator

# Initialize
authenticator = HyperspectralFaceAuthenticator(
    model_path='hyperspectral_embedding_model.h5',
    database_path='user_database.pkl',
    similarity_metric='cosine',
    threshold=0.6
)

# Enroll user
from face_authentication_system import load_hyperspectral_image
images = [
    load_hyperspectral_image('user1_img1.npy'),
    load_hyperspectral_image('user1_img2.npy'),
]
authenticator.enroll_user('user_001', images)

# Authenticate
test_image = load_hyperspectral_image('test.npy')
authenticated, matched_user, score = authenticator.authenticate(
    test_image,
    user_id='user_001'  # or None for 1:N identification
)

if authenticated:
    print(f"✓ Authenticated as {matched_user}")
else:
    print("✗ Authentication failed")
```

## Working with Real Data

### Dataset Format

The system expects hyperspectral face images in this format:

```
dataset/
├── subject_001/
│   ├── image_001.npy  # NumPy array: shape (H, W, C)
│   ├── image_002.npy
│   └── ...
├── subject_002/
│   └── ...
```

Where:
- **H, W**: Image height and width (default: 128×128)
- **C**: Number of spectral channels (default: 33)

### Preparing Your Data

If your data is in a different format:

```python
import numpy as np
from scipy.ndimage import zoom

# Load your hyperspectral image
# (implementation depends on your format)
img = load_your_data('image.ext')

# Resize to target size
target_size = (128, 128)
if img.shape[:2] != target_size:
    zoom_factors = (target_size[0]/img.shape[0], 
                   target_size[1]/img.shape[1], 1)
    img = zoom(img, zoom_factors, order=1)

# Normalize
img = (img - img.mean()) / (img.std() + 1e-7)

# Save as NumPy array
np.save('prepared_image.npy', img)
```

## Configuration Tips

### Choosing a Similarity Metric

**Cosine Similarity** (recommended):
```python
authenticator = HyperspectralFaceAuthenticator(
    similarity_metric='cosine',
    threshold=0.6  # Range: 0-1, higher = stricter
)
```

**Euclidean Distance**:
```python
authenticator = HyperspectralFaceAuthenticator(
    similarity_metric='euclidean',
    threshold=1.0  # Lower = stricter
)
```

### Tuning the Threshold

The threshold controls the tradeoff between security and convenience:

| Threshold | Security | User Experience | Use Case |
|-----------|----------|-----------------|----------|
| 0.3-0.4   | Low      | Very Convenient | Low-security applications |
| 0.5-0.6   | Medium   | Balanced        | General purpose |
| 0.7-0.8   | High     | Strict          | High-security applications |
| 0.9+      | Very High| May reject genuine users | Maximum security |

Test different thresholds:

```python
# Try multiple thresholds
for threshold in [0.5, 0.6, 0.7, 0.8]:
    authenticator.update_threshold(threshold)
    authenticated, _, score = authenticator.authenticate(test_image, user_id='alice')
    print(f"Threshold {threshold}: {'PASS' if authenticated else 'FAIL'} (score: {score:.4f})")
```

### Enrollment Best Practices

For best results when enrolling users:

1. **Use multiple images** (3-5 recommended)
   - Captures variations in lighting, pose, etc.
   - Improves authentication robustness

2. **Ensure good quality**
   - Proper alignment
   - Good lighting conditions
   - Minimal noise

3. **Capture variations**
   - Different facial expressions
   - Slight pose variations
   - Different acquisition times

```python
# Good enrollment example
images = [
    load_hyperspectral_image('user1_frontal.npy'),
    load_hyperspectral_image('user1_slight_left.npy'),
    load_hyperspectral_image('user1_slight_right.npy'),
]
authenticator.enroll_user('user_001', images)
```

## Testing Your Setup

Run the comprehensive test suite:

```bash
python test_system.py
```

This will verify:
- Module imports
- Model loading
- User enrollment
- 1:1 verification
- 1:N identification
- Database operations

## Troubleshooting

### Problem: "Model file not found"

**Solution**: Train the model first:
```bash
jupyter notebook hyperspectral_face_recognition_model.ipynb
```

### Problem: Low authentication accuracy

**Possible causes and solutions**:

1. **Threshold too high**
   - Try lowering the threshold
   - Test with values between 0.4-0.7

2. **Insufficient enrollment samples**
   - Enroll with 3-5 images per user
   - Ensure images have good quality

3. **Model not trained well**
   - Train for more epochs
   - Use more training data
   - Check training/validation curves

4. **Image preprocessing issues**
   - Verify images are normalized correctly
   - Check image dimensions match model input

### Problem: All users have similar scores

**Cause**: Model hasn't learned discriminative features

**Solutions**:
- Train with more diverse data
- Increase model capacity (more layers/units)
- Train for more epochs
- Use data augmentation

### Problem: Memory errors

**Solutions**:
- Reduce batch size in training
- Use smaller image dimensions
- Process fewer users at once

## Performance Optimization

### For Faster Training
```python
# Use GPU if available
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Increase batch size
history = model.fit(X_train, y_train, batch_size=32, ...)
```

### For Faster Authentication
```python
# Batch authenticate multiple users
embeddings = authenticator.model.predict(np.array(images), batch_size=32)
```

### For Production Deployment
- Consider model quantization
- Use TensorFlow Lite for edge devices
- Implement caching for frequently accessed users
- Use a proper database (Redis, MongoDB) instead of pickle

## Next Steps

1. **Collect Real Data**: Replace synthetic data with real hyperspectral faces
2. **Train Full Model**: Train with your dataset for production quality
3. **Tune Parameters**: Optimize threshold and model architecture for your use case
4. **Add Security**: Implement liveness detection and anti-spoofing
5. **Deploy**: Integrate into your application or service

## Additional Resources

- **README.md**: Complete system documentation
- **face_authentication_system.py**: API reference (see docstrings)
- **hyperspectral_face_recognition_model.ipynb**: Model architecture details
- **face_authentication_demo.ipynb**: Comprehensive usage examples

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the example notebooks
3. Open an issue on GitHub

---

**Note**: This system uses synthetic data for demonstration. Always test with real data before production deployment.
