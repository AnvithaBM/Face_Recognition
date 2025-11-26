# Face Authentication System

A complete face authentication system built on top of hyperspectral face recognition technology. This system provides user registration, authentication, and database management with support for both webcam and static images.

## Features

### 🎯 Core Capabilities
- **User Registration**: Capture and store facial features for new users
- **Face Authentication**: Compare captured faces against registered users
- **Unknown Face Rejection**: Properly reject unregistered users
- **Real-time Webcam Support**: Live face capture and processing
- **Persistent Storage**: JSON-based user database
- **Comprehensive Logging**: Track all authentication attempts

### 🛠️ Technical Features
- **Face Detection**: OpenCV Haar Cascade classifier
- **Feature Extraction**: CNN-based deep learning model
- **Similarity Matching**: Cosine similarity with configurable threshold
- **Preprocessing**: Support for RGB and hyperspectral images
- **Gabor Features**: Optional Gabor transform for enhanced recognition
- **Visualization**: PCA/t-SNE feature space visualization

## Installation

### Prerequisites
- Python 3.7 or higher
- Webcam (optional, for real-time capture)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages
- numpy
- pandas
- opencv-python
- Pillow
- tensorflow
- keras
- scikit-learn
- matplotlib
- seaborn

## Quick Start

### 1. Open the Notebook

```bash
jupyter notebook face_authentication_system.ipynb
```

### 2. Run Initial Setup Cells

Execute cells 1-11 to:
- Import libraries
- Configure system parameters
- Load or train the face recognition model
- Initialize feature extractor
- Set up user database

### 3. Register Users

**Option A: Using Sample Images (for testing)**
```python
# Create and register a sample user
sample_img = create_sample_face_image(person_id=1, variation=0)
register_new_user_from_image(sample_img, "Alice", feature_extractor, user_db)
```

**Option B: Using Your Own Image**
```python
# Load and register from your image
user_image = cv2.imread('path/to/your/photo.jpg')
register_new_user_from_image(user_image, "YourName", feature_extractor, user_db)
```

**Option C: Using Webcam**
```python
# Register using webcam capture
webcam_registration_workflow("YourName", feature_extractor, user_db)
```

### 4. Authenticate Users

**Option A: Using Sample Images**
```python
# Test authentication with sample
test_img = create_sample_face_image(person_id=1, variation=1)
authenticated, user_id, similarity = authenticate_from_image(
    test_img, feature_extractor, user_db
)
```

**Option B: Using Your Own Image**
```python
# Authenticate from your image
test_image = cv2.imread('path/to/test/photo.jpg')
authenticated, user_id, similarity = authenticate_from_image(
    test_image, feature_extractor, user_db
)
```

**Option C: Using Webcam**
```python
# Authenticate using webcam
authenticated, user_id, similarity = webcam_authentication_workflow(
    feature_extractor, user_db
)
```

## System Architecture

### Components

1. **Face Detection Module**
   - Uses OpenCV Haar Cascade
   - Detects faces in images/video frames
   - Extracts face regions with padding

2. **Preprocessing Module**
   - Resizes images to 128x128
   - Normalizes pixel values
   - Applies Gabor transform (optional)
   - Handles RGB and hyperspectral formats

3. **Feature Extraction Module**
   - CNN-based deep learning model
   - Extracts 256-dimensional feature vectors
   - Uses pre-trained model layers

4. **Authentication Module**
   - Cosine similarity calculation
   - Threshold-based matching
   - Multi-user comparison

5. **Database Module**
   - JSON-based storage
   - User metadata management
   - Feature vector persistence

## Configuration

### Key Parameters

```python
# Image size
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Authentication threshold (0-1)
# Lower = more lenient, Higher = more strict
SIMILARITY_THRESHOLD = 0.6

# Gabor transform
USE_GABOR = True

# Database files
USER_DB_PATH = 'user_database.json'
MODEL_PATH = 'face_recognition_model.h5'
LOG_FILE = 'authentication_log.txt'
```

### Adjusting Sensitivity

```python
# More strict (fewer false positives, may reject legitimate users)
authenticate_from_image(image, feature_extractor, user_db, threshold=0.8)

# More lenient (more false positives, fewer rejections)
authenticate_from_image(image, feature_extractor, user_db, threshold=0.5)
```

## File Structure

```
Face_Recognition/
├── face_authentication_system.ipynb    # Main authentication notebook
├── face_recognition_hyperspectral (3).ipynb  # Original training notebook
├── requirements.txt                     # Python dependencies
├── AUTHENTICATION_SYSTEM_README.md     # This file
├── user_database.json                   # User database (created at runtime)
├── face_recognition_model.h5           # Trained model (created at runtime)
└── authentication_log.txt              # Authentication logs (created at runtime)
```

## Database Management

### List Registered Users
```python
print("Registered users:", user_db.get_all_users())
```

### Get User Information
```python
user_info = user_db.get_user("UserName")
print(user_info)
```

### Delete a User
```python
user_db.delete_user("UserName")
```

### Backup Database
```bash
cp user_database.json user_database_backup.json
```

## Visualization and Analysis

### View Feature Space
```python
# Visualize user features in 2D space
visualize_user_features(user_db, method='pca')
```

### Similarity Matrix
```python
# Plot similarity between all users
plot_similarity_matrix(user_db)
```

## Troubleshooting

### Webcam Issues

**Problem**: Webcam not accessible in notebook environment

**Solution**: Use image files instead
```python
# Register with image file
img = cv2.imread('your_photo.jpg')
register_new_user_from_image(img, "YourName", feature_extractor, user_db)
```

### No Face Detected

**Problem**: "No face detected in image"

**Solutions**:
1. Ensure face is clearly visible and well-lit
2. Face should be front-facing
3. Reduce `scale_factor` in `detect_faces()` function
4. Check image is not corrupted

### Low Authentication Accuracy

**Problem**: Legitimate users not authenticated

**Solutions**:
1. Lower similarity threshold (e.g., 0.5 instead of 0.6)
2. Register multiple samples per user
3. Ensure consistent lighting conditions
4. Use higher quality images

### Database Corruption

**Problem**: Error loading user database

**Solution**: Delete and recreate database
```bash
rm user_database.json
# Re-register users
```

## Performance Considerations

### Speed Optimization
- **Small databases (<100 users)**: Current implementation is sufficient
- **Large databases (>1000 users)**: Consider using FAISS for approximate nearest neighbor search

### Memory Usage
- Each user stores a ~256-dimensional feature vector
- 1000 users ≈ 1-2 MB of storage

### Accuracy vs Speed
- Higher thresholds = fewer comparisons needed = faster
- Lower thresholds = more thorough checking = slower

## Security Considerations

### Current Implementation
- Basic face recognition (no liveness detection)
- Vulnerable to photo/video replay attacks
- No encryption of stored features

### Recommendations for Production
1. Add liveness detection (blink detection, movement)
2. Encrypt user database
3. Implement rate limiting for authentication attempts
4. Add multi-factor authentication
5. Use secure storage (not plain JSON)
6. Implement access control and audit logging

## Limitations

1. **Photo Spoofing**: Cannot distinguish between real face and photo
2. **Lighting Sensitivity**: Performance degrades in poor lighting
3. **Angle Sensitivity**: Best results with front-facing images
4. **Expression Variations**: Large expression changes may affect accuracy
5. **Age Changes**: Features may drift over time (requires re-registration)

## Future Enhancements

### Planned Features
- [ ] Multi-sample registration (store multiple faces per user)
- [ ] Continuous authentication from video stream
- [ ] Face anti-spoofing / liveness detection
- [ ] Database migration to vector database (FAISS)
- [ ] Face alignment using facial landmarks
- [ ] Model fine-tuning with registered users
- [ ] Web-based GUI interface
- [ ] Mobile app integration
- [ ] Cloud deployment support

## Testing

The notebook includes comprehensive tests:
1. **User Registration Tests**: Register sample users (Alice, Bob, Charlie)
2. **Known User Tests**: Authenticate registered users with variations
3. **Unknown User Tests**: Reject unregistered users
4. **System Statistics**: View database status and logs

Run cells 26-29 to execute all tests.

## API Reference

### Registration Functions

```python
register_new_user_from_image(image, user_id, feature_extractor, user_db, visualize=True)
"""
Register a new user from image.

Args:
    image: Input image (BGR format)
    user_id: Unique user identifier (string)
    feature_extractor: Feature extraction model
    user_db: UserDatabase instance
    visualize: Show result visualization

Returns:
    Boolean indicating success
"""
```

### Authentication Functions

```python
authenticate_from_image(image, feature_extractor, user_db, threshold=0.6, visualize=True)
"""
Authenticate user from image.

Args:
    image: Input image (BGR format)
    feature_extractor: Feature extraction model
    user_db: UserDatabase instance
    threshold: Similarity threshold (0-1)
    visualize: Show result visualization

Returns:
    tuple: (authenticated, user_id, similarity_score)
"""
```

### Database Functions

```python
user_db.register_user(user_id, features, metadata=None)
user_db.get_user(user_id)
user_db.get_all_users()
user_db.delete_user(user_id)
user_db.get_all_features()
```

## Contributing

Contributions are welcome! Areas for improvement:
- Enhanced face detection algorithms
- Better feature extraction methods
- Liveness detection implementation
- Performance optimizations
- Documentation improvements

## License

This project is part of the Face_Recognition repository.

## References

- OpenCV Documentation: https://docs.opencv.org/
- TensorFlow: https://www.tensorflow.org/
- Face Recognition Research: https://arxiv.org/abs/1804.06655

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the notebook comments and documentation
3. Open an issue in the repository

---

**Note**: This system is designed for educational and demonstration purposes. For production use, implement additional security measures and testing.
