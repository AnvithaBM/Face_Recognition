# Face Recognition Authentication System

A real-time face authentication system using VGG-based deep learning model with Gabor filters for robust face recognition.

## Features

- **Real-time User Registration**: Capture multiple face images via camera to register new users
- **Face Authentication**: Verify user identity using face recognition
- **Web-based UI**: Modern, responsive interface built with Flask and HTML/CSS/JavaScript
- **Feature Extraction**: Uses VGG model to extract facial features (embeddings)
- **Similarity Matching**: Employs cosine similarity for accurate face matching
- **Gabor Transform**: Applies Gabor filters for enhanced feature extraction from RGB images

## System Requirements

- Python 3.8+
- Webcam/Camera for capturing images
- Modern web browser with camera access support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AnvithaBM/Face_Recognition.git
cd Face_Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the trained model file:
- Place `best_model.keras` in the root directory
- The model should be trained using the provided Jupyter notebook

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. **Register a New User**:
   - Click "Register" on the home page
   - Enter a unique username
   - Click "Start Camera" to enable webcam
   - Capture 5-10 face images from different angles
   - Click "Register User" to save

4. **Authenticate**:
   - Click "Authenticate" on the home page
   - Click "Start Camera"
   - Position your face in the frame
   - Click "Authenticate" to verify identity

## Technical Details

### Model Architecture
- Base: VGG-inspired CNN architecture
- Input: 128x128x3 Gabor-transformed images
- Feature Extraction: Penultimate layer outputs (256-dimensional embeddings)
- Matching: Cosine similarity with threshold of 0.7

### Gabor Transform Parameters
- Kernel size: 31x31
- Sigma: 4.0
- Orientations: 4 (0°, 45°, 90°, 135°)
- Lambda: 10.0
- Gamma: 0.5

### Data Storage
- User embeddings stored in `user_features.json`
- Each user has averaged features from multiple registration images

## Project Structure

```
Face_Recognition/
├── app.py                          # Flask application
├── model_utils.py                  # Model utilities (alternative implementation)
├── face_auth.py                    # Face authentication module (alternative)
├── best_model.keras                # Trained VGG model
├── requirements.txt                # Python dependencies
├── templates/
│   ├── index.html                 # Landing page
│   ├── register.html              # Registration page
│   └── authenticate.html          # Authentication page
├── static/
│   ├── css/
│   │   └── style.css             # Stylesheet
│   └── js/
│       ├── register.js           # Registration logic
│       └── authenticate.js       # Authentication logic
└── face_recognition_hyperspectral (3).ipynb  # Model training notebook
```

## Security Considerations

- User features stored locally in JSON format
- For production: Use database with encryption
- Implement HTTPS for secure communication
- Add user authentication and access control
- Consider adding liveness detection to prevent spoofing

## Future Enhancements

- Add database support (PostgreSQL/MongoDB)
- Implement user management (delete, update)
- Add liveness detection
- Support multiple face angles during authentication
- Add logging and monitoring
- Deploy to cloud platform

## Credits

Final Year Project - Face Recognition System
Based on VGG architecture with Gabor filters for hyperspectral face recognition

## License

This project is for educational purposes.