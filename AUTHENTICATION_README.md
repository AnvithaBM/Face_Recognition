# Face Authentication System

A complete face authentication application based on hyperspectral face recognition with Gabor transform. This system provides user registration, authentication (1:N identification), and verification (1:1 matching) capabilities through a user-friendly Streamlit web interface.

## Features

- **Feature Extraction**: Uses a trained CNN model to generate face embeddings/templates
- **User Registration**: Register new users with multiple face samples for robust templates
- **Authentication (1:N)**: Identify users from the entire database
- **Verification (1:1)**: Verify if a face matches a specific registered user
- **Gabor Transform**: Enhanced feature extraction using Gabor filters for hyperspectral images
- **Web Interface**: Intuitive Streamlit-based UI with separate tabs for different operations
- **Database Management**: Persistent storage using pickle with export capabilities
- **Configurable Threshold**: Adjustable similarity threshold for authentication
- **Multi-Sample Averaging**: Robust registration using multiple face samples

## Project Structure

```
Face_Recognition/
├── app.py                          # Streamlit web application
├── face_authentication.py          # Core authentication system
├── feature_extractor.py            # Feature extraction module
├── utils.py                        # Helper functions
├── requirements.txt                # Python dependencies
├── AUTHENTICATION_README.md        # This file
├── face_recognition_hyperspectral (3).ipynb  # Original notebook
└── best_model.h5                   # Trained model (if available)
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AnvithaBM/Face_Recognition.git
   cd Face_Recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Place trained model**
   - If you have a trained model, place it as `best_model.h5` or `hyperspectral_face_recognition_model.keras`
   - If no model is available, the system will use a dummy model for demonstration

## Usage

### Running the Web Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Web Interface Sections

#### 1. User Registration Tab
- Enter a unique User ID
- Optionally add metadata (name, email, department)
- Upload 2-5 face images of the same person
- Click "Register User" to complete registration

#### 2. Authentication Tab (1:N Identification)
- Upload a face image
- Click "Authenticate" to identify the user from the database
- View confidence score and user information

#### 3. Verification Tab (1:1 Matching)
- Select a user from the dropdown
- Upload a face image
- Click "Verify" to check if the image matches the selected user

#### Settings Sidebar
- Adjust authentication threshold (0.0 - 1.0)
- View system statistics
- List all registered users
- Export database to JSON

### Using the Python API

#### Basic Usage

```python
from face_authentication import FaceAuthenticationSystem
from PIL import Image

# Initialize the system
auth_system = FaceAuthenticationSystem(
    model_path='best_model.h5',
    similarity_threshold=0.6,
    use_gabor=True
)

# Register a user
images = [Image.open('user1_sample1.jpg'), Image.open('user1_sample2.jpg')]
success, message = auth_system.register_user(
    user_id='john_doe',
    images=images,
    metadata={'name': 'John Doe', 'department': 'Engineering'}
)
print(message)

# Authenticate (identify from database)
test_image = Image.open('test_face.jpg')
user_id, confidence, message = auth_system.authenticate_user(test_image)
print(f"Identified: {user_id} with confidence {confidence:.2%}")

# Verify (check specific user)
verified, similarity, message = auth_system.verify_user('john_doe', test_image)
print(f"Verification: {'Success' if verified else 'Failed'}")
```

#### Advanced Features

```python
# List all users
users = auth_system.list_users()
for user in users:
    print(f"{user['user_id']}: {user['num_samples']} samples")

# Get user information
info = auth_system.get_user_info('john_doe')
print(info)

# Update user
success, message = auth_system.update_user('john_doe', new_images)

# Delete user
success, message = auth_system.delete_user('john_doe')

# Export database
auth_system.export_database('backup.json', format='json')

# Get system statistics
stats = auth_system.get_statistics()
print(stats)
```

## Architecture

### Image Processing Pipeline

1. **Input**: RGB/Grayscale/Hyperspectral image
2. **Preprocessing**: 
   - Resize to 128x128
   - Normalize to [0, 1]
   - Apply Gabor transform (optional)
3. **Feature Extraction**: CNN model generates embedding vector
4. **Matching**: Cosine similarity between embeddings
5. **Decision**: Compare similarity with threshold

### Model Architecture

The system uses a custom CNN architecture with:
- 4 convolutional blocks (32, 64, 128, 256 filters)
- Batch normalization and dropout for regularization
- Dense layers (512, 256) for feature extraction
- Supports 4-channel input (RGB + Gabor)

Note: If using the original notebook model, train it using the Jupyter notebook provided.

### Gabor Transform

The Gabor transform enhances face features by:
- Extracting texture information at specific orientations
- Improving robustness to lighting variations
- Adding spectral information to spatial features

Configuration (in `utils.py`):
- Kernel size: 31x31
- Sigma: 4.0
- Theta: π/4
- Lambda: 10.0
- Gamma: 0.5

## Configuration

### Similarity Threshold

The threshold determines how similar embeddings must be for authentication:
- **0.5-0.6**: Moderate security, better for varied lighting
- **0.6-0.7**: Balanced (recommended)
- **0.7-0.8**: High security, may reject genuine users in poor conditions
- **0.8-1.0**: Very strict, only for controlled environments

Adjust via:
- Web UI: Settings sidebar slider
- Python API: `auth_system.set_threshold(0.65)`

### Image Requirements

For best results:
- **Resolution**: Minimum 128x128 pixels
- **Quality**: Clear, well-lit face images
- **Format**: PNG, JPG, JPEG, BMP
- **Face**: Frontal or near-frontal view
- **Multiple Samples**: 3-5 images per user for robust registration

## Troubleshooting

### Model Not Found
If you see "No trained model found" warning:
- The system will use a dummy model for demonstration
- Train a model using the provided Jupyter notebook
- Place the trained model as `best_model.h5`

### Low Confidence Scores
If authentication confidence is consistently low:
- Lower the threshold in settings
- Register users with more sample images
- Ensure images are well-lit and clear
- Verify images are properly preprocessed

### Database Issues
If registration/authentication fails:
- Check file permissions for `face_database.pkl`
- Delete `face_database.pkl` to start fresh
- Ensure sufficient disk space

## Security Considerations

1. **Template Storage**: Face embeddings are stored in plaintext pickle files. For production:
   - Use encrypted storage
   - Implement access controls
   - Add audit logging

2. **Liveness Detection**: This system does not include anti-spoofing. For production:
   - Add liveness detection
   - Implement challenge-response
   - Use multiple authentication factors

3. **Privacy**: Face biometric data is sensitive:
   - Comply with GDPR/privacy regulations
   - Implement data retention policies
   - Provide user consent mechanisms

## Performance

Typical performance metrics:
- **Feature Extraction**: ~50-200ms per image (CPU)
- **Registration**: ~2-5 seconds for 3 images
- **Authentication**: ~100-500ms (depends on database size)
- **Memory**: ~100MB for model + ~1KB per registered user

## Future Enhancements

Potential improvements:
- [ ] Real-time face detection and alignment
- [ ] Liveness detection / anti-spoofing
- [ ] Support for face masks/accessories
- [ ] Age invariant recognition
- [ ] Multi-face processing in single image
- [ ] GPU acceleration support
- [ ] REST API for integration
- [ ] Mobile app support

## References

- Original notebook: `face_recognition_hyperspectral (3).ipynb`
- Gabor Filters: See `Gabor_CNN.pdf`
- Dataset: UWA HSFD V1.1 (Hyperspectral Face Database)

## License

This project follows the repository license.

## Author

Developed as an extension of the hyperspectral face recognition notebook.

## Support

For issues or questions:
1. Check this README
2. Review the Jupyter notebook for model training
3. Open an issue on GitHub
