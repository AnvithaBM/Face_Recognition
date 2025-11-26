# Face Recognition System

This repository contains hyperspectral face recognition and authentication systems.

## Contents

### 1. Hyperspectral Face Recognition
- **File**: `face_recognition_hyperspectral (3).ipynb`
- **Description**: Training notebook for hyperspectral face recognition using CNN
- **Features**:
  - Hyperspectral image processing
  - Gabor feature extraction
  - CNN-based classification
  - Model training and evaluation

### 2. Face Authentication System (NEW)
- **File**: `face_authentication_system.ipynb`
- **Documentation**: `AUTHENTICATION_SYSTEM_README.md`
- **Description**: Complete face authentication system with user registration and verification
- **Features**:
  - User registration from images/webcam
  - Real-time face authentication
  - Unknown face rejection
  - Persistent user database
  - Similarity-based matching
  - Comprehensive logging

## Quick Start

### Face Authentication System

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Open the notebook:
```bash
jupyter notebook face_authentication_system.ipynb
```

3. Run all cells to initialize the system

4. Register users and test authentication

For detailed instructions, see [AUTHENTICATION_SYSTEM_README.md](AUTHENTICATION_SYSTEM_README.md)

## System Requirements

- Python 3.7+
- TensorFlow 2.10+
- OpenCV 4.5+
- Webcam (optional, for real-time capture)

## Documentation

- [Authentication System Guide](AUTHENTICATION_SYSTEM_README.md) - Complete guide for the face authentication system
- [Requirements](requirements.txt) - Python package dependencies

## Features

✅ Face detection using OpenCV Haar Cascades
✅ CNN-based feature extraction
✅ User registration and management
✅ Cosine similarity authentication
✅ Real-time webcam integration
✅ JSON-based user database
✅ Authentication logging
✅ Visualization tools

## Use Cases

- Access control systems
- User authentication
- Identity verification
- Security systems
- Attendance tracking

## License

See repository license.