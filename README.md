# Hyperspectral Face Recognition and Authentication System

A complete face recognition and authentication system using hyperspectral imaging data. This system provides both model training and a production-ready authentication pipeline with user enrollment and verification capabilities.

## Features

### Core Capabilities
- **Hyperspectral Face Recognition**: CNN-based model for extracting facial embeddings from hyperspectral images
- **User Enrollment**: Register new users with multiple face samples for robust authentication
- **1:1 Verification**: Verify a specific user's claimed identity
- **1:N Identification**: Identify users from a database without prior claim
- **Flexible Similarity Metrics**: Support for cosine similarity and Euclidean distance
- **Threshold-based Authentication**: Configurable acceptance/rejection thresholds
- **Database Management**: Add, remove, export, and import user data
- **Modular Design**: Easy integration into existing authentication pipelines

### Technical Highlights
- L2-normalized embeddings for robust similarity computation
- Support for UWA HSFD dataset structure
- Pickle-based user database for simplicity
- TensorFlow/Keras implementation
- Both notebook and script interfaces

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/AnvithaBM/Face_Recognition.git
cd Face_Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Model

First, train the hyperspectral face recognition model using the provided notebook:

```bash
jupyter notebook hyperspectral_face_recognition_model.ipynb
```

This will:
- Load or generate hyperspectral face data
- Train a CNN-based embedding model
- Save the trained model as `hyperspectral_embedding_model.h5`
- Generate training metrics and visualizations

**Note**: The notebook includes synthetic data generation for demonstration. For production use, replace with real hyperspectral face images.

### 2. Run the Authentication Demo

Try the complete authentication system with synthetic data:

```bash
python example_usage.py --demo
```

Or explore the interactive demo notebook:

```bash
jupyter notebook face_authentication_demo.ipynb
```

## Usage

### Command Line Interface

#### Enroll a New User
```bash
python example_usage.py --enroll alice --images face1.npy face2.npy face3.npy
```

#### Authenticate (1:1 Verification)
```bash
# Verify that the image is of Alice
python example_usage.py --authenticate test_face.npy --user alice
```

#### Identify (1:N Identification)
```bash
# Find which user this image belongs to
python example_usage.py --authenticate test_face.npy
```

#### List Users
```bash
python example_usage.py --list
```

#### Remove User
```bash
python example_usage.py --remove alice
```

#### Adjust Threshold
```bash
# Higher threshold = stricter authentication
python example_usage.py --authenticate test.npy --threshold 0.8
```

### Python API

```python
from face_authentication_system import HyperspectralFaceAuthenticator, load_hyperspectral_image

# Initialize the authenticator
authenticator = HyperspectralFaceAuthenticator(
    model_path='hyperspectral_embedding_model.h5',
    database_path='user_database.pkl',
    similarity_metric='cosine',
    threshold=0.6
)

# Enroll a user
images = [
    load_hyperspectral_image('user1_img1.npy'),
    load_hyperspectral_image('user1_img2.npy'),
    load_hyperspectral_image('user1_img3.npy')
]
authenticator.enroll_user('user_001', images)

# Authenticate (1:1 verification)
test_image = load_hyperspectral_image('test.npy')
authenticated, matched_user, score = authenticator.authenticate(
    test_image, 
    user_id='user_001'
)

if authenticated:
    print(f"✓ User verified: {matched_user} (score: {score:.4f})")
else:
    print(f"✗ Authentication failed (score: {score:.4f})")

# Identify (1:N identification)
authenticated, matched_user, score = authenticator.authenticate(
    test_image,
    user_id=None  # Search all users
)

if authenticated:
    print(f"✓ User identified: {matched_user} (score: {score:.4f})")
else:
    print(f"✗ No match found (best score: {score:.4f})")
```

## Dataset Structure

The system is designed for the UWA HSFD (Hyperspectral Face Dataset) structure:

```
dataset/
├── subject_001/
│   ├── image_001.npy
│   ├── image_002.npy
│   └── ...
├── subject_002/
│   ├── image_001.npy
│   └── ...
└── ...
```

Hyperspectral images should be stored as:
- **NumPy arrays** (`.npy`): Shape `(H, W, C)` where C is the number of spectral channels
- **MATLAB files** (`.mat`): With key `hyperspectral_image`

Default specifications:
- Image size: 128×128 pixels
- Spectral channels: 33 (typical for UWA HSFD)
- Normalized per-image (mean=0, std=1)

## System Architecture

### Model Architecture

The hyperspectral face recognition model consists of:

1. **Feature Extraction**: 4 convolutional blocks with batch normalization and dropout
   - Conv2D(32) → MaxPool → Dropout
   - Conv2D(64) → MaxPool → Dropout
   - Conv2D(128) → MaxPool → Dropout
   - Conv2D(256) → MaxPool → Dropout

2. **Embedding Layer**: Dense(512) → Dense(128) with L2 normalization
   - 128-dimensional embeddings
   - L2 normalized for robust similarity computation

3. **Classification Head**: Dense(num_classes) with softmax (for training only)

### Authentication Pipeline

```
Input Image → Preprocessing → Embedding Extraction → Similarity Computation → Threshold Check → Accept/Reject
```

**Preprocessing**: Resize, normalize
**Embedding**: Extract 128-D feature vector
**Similarity**: Cosine similarity or Euclidean distance
**Threshold**: Configurable acceptance threshold

## Configuration

### Similarity Metrics

**Cosine Similarity** (default):
- Range: [-1, 1], where 1 = identical
- Recommended threshold: 0.5-0.8
- Better for normalized embeddings

**Euclidean Distance**:
- Range: [0, ∞], where 0 = identical
- Recommended threshold: 0.5-2.0 (inverted for internal use)
- Better for absolute differences

### Threshold Tuning

The authentication threshold balances security and convenience:

| Threshold | Security | False Rejection | False Acceptance |
|-----------|----------|-----------------|------------------|
| 0.3       | Low      | Low             | High             |
| 0.5       | Medium   | Medium          | Medium           |
| 0.7       | High     | High            | Low              |
| 0.9       | Very High| Very High       | Very Low         |

**Recommendation**: Start with 0.6 and adjust based on your use case.

## Files and Structure

```
Face_Recognition/
├── README.md                                      # This file
├── requirements.txt                               # Python dependencies
├── hyperspectral_face_recognition_model.ipynb    # Model training notebook
├── face_authentication_system.py                 # Authentication system module
├── face_authentication_demo.ipynb                # Interactive demo notebook
├── example_usage.py                              # Command-line interface
├── hyperspectral_embedding_model.h5              # Trained model (generated)
├── user_database.pkl                             # User database (generated)
└── model_metadata.pkl                            # Model metadata (generated)
```

## Advanced Usage

### Custom Dataset

To use your own hyperspectral face dataset:

1. Organize data in the expected structure (see Dataset Structure)
2. Update the `DATA_PATH` in the training notebook
3. Adjust `img_size` and `num_channels` if needed
4. Run the training notebook
5. Use the generated model for authentication

### Batch Enrollment

```python
# Enroll multiple users from a dataset
import os
from face_authentication_system import load_hyperspectral_image

dataset_path = './UWA_HSFD_dataset'

for subject_dir in os.listdir(dataset_path):
    subject_path = os.path.join(dataset_path, subject_dir)
    if not os.path.isdir(subject_path):
        continue
    
    # Load all images for this subject
    images = []
    for img_file in os.listdir(subject_path):
        if img_file.endswith('.npy'):
            img_path = os.path.join(subject_path, img_file)
            images.append(load_hyperspectral_image(img_path))
    
    # Enroll user
    authenticator.enroll_user(subject_dir, images)
```

### Database Export/Import

```python
# Export database for backup or sharing
authenticator.export_database('backup_2024.pkl')

# Import database
authenticator.import_database('backup_2024.pkl', merge=True)
```

### Integration Example

```python
def secure_access_control(user_claim: str, face_image_path: str):
    """Example integration for access control system."""
    
    # Load user's face image
    face_image = load_hyperspectral_image(face_image_path)
    
    # Authenticate
    authenticated, matched_user, confidence = authenticator.authenticate(
        face_image,
        user_id=user_claim
    )
    
    if authenticated and matched_user == user_claim:
        # Grant access
        log_access(user_claim, "GRANTED", confidence)
        return True
    else:
        # Deny access
        log_access(user_claim, "DENIED", confidence)
        return False
```

## Performance

Typical performance on modern hardware:

- **Training**: ~10-30 minutes (50 epochs, synthetic data)
- **Enrollment**: ~100-200ms per user (3 images)
- **Authentication**: ~50-100ms per query
- **Throughput**: ~10-20 authentications/second

## Limitations and Future Work

### Current Limitations
- Synthetic data used for demonstration (replace with real hyperspectral images)
- Simple pickle-based database (consider SQL/NoSQL for production)
- No liveness detection (vulnerable to photo attacks)
- No anti-spoofing mechanisms
- Single model checkpoint (no ensemble)

### Future Enhancements
- [ ] Liveness detection integration
- [ ] Anti-spoofing with presentation attack detection
- [ ] Real-time video authentication
- [ ] Distributed database support (Redis, MongoDB)
- [ ] Model quantization for edge deployment
- [ ] Multi-factor authentication integration
- [ ] Encrypted database storage
- [ ] API server with REST/gRPC endpoints
- [ ] Web interface for administration

## Troubleshooting

### Model File Not Found
```
FileNotFoundError: Model file not found: hyperspectral_embedding_model.h5
```
**Solution**: Train the model first using `hyperspectral_face_recognition_model.ipynb`

### Low Authentication Accuracy
**Possible causes**:
- Threshold too high/low
- Insufficient enrollment samples
- Image quality issues
- Model not trained sufficiently

**Solutions**:
- Adjust threshold (try different values)
- Enroll with more images (3-5 recommended)
- Verify image preprocessing
- Retrain with more data/epochs

### Memory Issues
**Solution**: Reduce batch size in training or use smaller image dimensions

## Citation

If you use this system in your research, please cite:

```bibtex
@software{hyperspectral_face_auth,
  title = {Hyperspectral Face Recognition and Authentication System},
  author = {AnvithaBM},
  year = {2025},
  url = {https://github.com/AnvithaBM/Face_Recognition}
}
```

## License

This project is available for research and educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This system uses synthetic data for demonstration. For production deployment, use real hyperspectral face data and implement additional security measures.