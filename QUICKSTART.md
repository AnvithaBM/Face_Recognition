# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the Web Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Running the Demo

```bash
# Run demo with synthetic data
python3 demo.py
```

This will:
- Create 3 synthetic users
- Register them with multiple samples
- Test authentication (1:N identification)
- Test verification (1:1 matching)
- Export database

## Using the Python API

### Basic Example

```python
from face_authentication import FaceAuthenticationSystem
from PIL import Image

# Initialize
auth_system = FaceAuthenticationSystem(
    model_path='best_model.h5',  # or 'hyperspectral_face_recognition_model.keras'
    similarity_threshold=0.6,
    use_gabor=True
)

# Register a user
images = [
    Image.open('user1_photo1.jpg'),
    Image.open('user1_photo2.jpg'),
    Image.open('user1_photo3.jpg')
]

success, message = auth_system.register_user(
    user_id='john_doe',
    images=images,
    metadata={'name': 'John Doe', 'department': 'Engineering'}
)
print(message)

# Authenticate (identify from database)
test_image = Image.open('unknown_person.jpg')
user_id, confidence, message = auth_system.authenticate_user(test_image)

if user_id:
    print(f"Identified as: {user_id}")
    print(f"Confidence: {confidence*100:.1f}%")
else:
    print("Authentication failed")

# Verify (check specific user)
verified, similarity, message = auth_system.verify_user('john_doe', test_image)

if verified:
    print(f"Verified! Similarity: {similarity*100:.1f}%")
else:
    print("Verification failed")
```

### Advanced Features

```python
# List all registered users
users = auth_system.list_users()
for user in users:
    print(f"{user['user_id']}: {user['num_samples']} samples")

# Get specific user info
info = auth_system.get_user_info('john_doe')
print(info)

# Update user with new images
new_images = [Image.open('user1_new1.jpg'), Image.open('user1_new2.jpg')]
success, message = auth_system.update_user('john_doe', new_images)

# Delete user
success, message = auth_system.delete_user('john_doe')

# Change threshold
auth_system.set_threshold(0.7)  # More strict

# Export database
auth_system.export_database('backup.json', format='json')

# Get statistics
stats = auth_system.get_statistics()
print(f"Total users: {stats['total_users']}")
print(f"Feature dimension: {stats['feature_dimension']}")
```

## Training Your Own Model

To train a model on the hyperspectral dataset:

1. Open `face_recognition_hyperspectral (3).ipynb` in Jupyter
2. Set your dataset path
3. Run all cells to train the model
4. The model will be saved as `best_face_recognition_model.keras`
5. Use this model file with the authentication system

## Configuration

### Gabor Transform Settings

Edit `utils.py` to customize Gabor parameters:

```python
GABOR_KSIZE = 31     # Kernel size
GABOR_SIGMA = 4.0    # Standard deviation
GABOR_THETA = np.pi / 4  # Orientation
GABOR_LAMBDA = 10.0  # Wavelength
GABOR_GAMMA = 0.5    # Spatial aspect ratio
```

### Similarity Threshold

Adjust in code or web UI:
- **0.5-0.6**: More lenient (may have false positives)
- **0.6-0.7**: Balanced (recommended)
- **0.7-0.8**: Strict (may reject genuine users)
- **0.8-1.0**: Very strict (only for controlled environments)

## Troubleshooting

### "No trained model found"
- The system will use a dummy model for demonstration
- To use a real model, train it using the notebook or place a trained model file

### Low confidence scores
- Lower the threshold
- Use better quality images
- Register with more samples (3-5 images)

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### Database issues
- Delete `face_database.pkl` to start fresh
- Check file permissions

## Project Structure

```
Face_Recognition/
├── app.py                          # Streamlit web app
├── face_authentication.py          # Core authentication
├── feature_extractor.py            # Feature extraction
├── utils.py                        # Helper functions
├── demo.py                         # Demo script
├── requirements.txt                # Dependencies
├── AUTHENTICATION_README.md        # Full documentation
├── QUICKSTART.md                   # This file
└── face_recognition_hyperspectral (3).ipynb  # Training notebook
```

## Next Steps

1. **For Demo**: Run `python3 demo.py` to see the system in action
2. **For Web UI**: Run `streamlit run app.py` to use the interface
3. **For Development**: Read `AUTHENTICATION_README.md` for detailed docs
4. **For Training**: Use the Jupyter notebook to train on your dataset

## Support

For issues or questions:
- Check `AUTHENTICATION_README.md` for detailed documentation
- Review the demo script for usage examples
- Check the Jupyter notebook for model training details
