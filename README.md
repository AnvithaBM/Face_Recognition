# Face Recognition Authentication System

A robust face authentication web application using Gabor Transform and Deep Learning (CNN). The system allows user registration by capturing face images via webcam, applying Gabor transform for texture feature extraction, and using a trained CNN model to extract deep features for authentication.

## Features

- **Face Registration**: Capture multiple face images to register a new user
- **Face Authentication**: Verify user identity using face recognition
- **Gabor Transform**: Advanced texture feature extraction using Gabor filters
- **Deep Learning**: CNN model for extracting 512-dimensional feature vectors
- **Cosine Similarity Matching**: Robust face matching with confidence scores
- **Web Interface**: User-friendly Flask web application with webcam access
- **Real-time Processing**: Live webcam feed for image capture

## Architecture

1. **Gabor Transform**: Input face images are processed using Gabor filters at multiple orientations to extract texture features
2. **Feature Extraction**: A trained CNN model extracts 512-dimensional feature vectors from the Gabor-transformed images
3. **Storage**: User features are stored in JSON format with averaged features from multiple samples
4. **Authentication**: Cosine similarity is used to match captured face features against registered users (threshold: 0.7)

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam for image capture
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/AnvithaBM/Face_Recognition.git
cd Face_Recognition
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

Before running the web application, you need to train the face recognition model:

1. Open the Jupyter notebook `face_recognition_hyperspectral (3).ipynb`
2. Run all cells to train the model on your dataset
3. The trained model will be saved as `best_face_recognition_model.keras` or `best_model.h5`
4. If the model is saved as `.keras`, you can rename it to `best_model.h5` for consistency:
   ```bash
   mv best_face_recognition_model.keras best_model.h5
   ```

### Step 2: Create Feature Extractor

After training the model, create the feature extractor that outputs 512-dimensional features:

```bash
python save_model.py
```

This script will:
- Load the trained model (`best_model.h5`)
- Extract the Dense(512) layer to create a feature extractor
- Save the feature extractor as `feature_extractor.h5`

### Step 3: Run the Web Application

Start the Flask web server:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Step 4: Use the Application

#### Registration
1. Navigate to `http://localhost:5000`
2. Click "Go to Registration"
3. Enter a unique User ID
4. Click "Start Camera" to access your webcam
5. Capture 3-5 face images (vary your pose slightly between captures)
6. Click "Register User"

#### Authentication
1. Navigate to `http://localhost:5000`
2. Click "Go to Authentication"
3. Click "Start Camera" to access your webcam
4. Position your face in the center of the frame
5. Click "Authenticate"
6. View your authentication result with confidence score

## File Structure

```
Face_Recognition/
├── app.py                          # Flask web application
├── save_model.py                   # Script to create feature extractor
├── user_data.json                  # User database (registered features)
├── requirements.txt                # Python dependencies
├── face_recognition_hyperspectral (3).ipynb  # Training notebook
├── templates/
│   ├── index.html                  # Home page
│   ├── register.html               # Registration page
│   └── authenticate.html           # Authentication page
├── static/
│   ├── js/
│   │   └── camera.js              # Webcam and capture logic
│   └── css/
│       └── style.css              # Styling
└── README.md                       # This file
```

## Model Architecture

The CNN model uses a custom architecture with:
- 4 Convolutional blocks (32, 64, 128, 256 filters)
- Batch Normalization and Dropout for regularization
- Dense layers (512, 256) for feature extraction
- Softmax output for classification during training

For authentication, we extract the 512-dimensional features from the Dense(512) layer.

## Configuration

Key parameters in `app.py`:

- `SIMILARITY_THRESHOLD`: 0.7 (minimum cosine similarity for successful authentication)
- `IMG_HEIGHT`: 128 pixels
- `IMG_WIDTH`: 128 pixels
- Gabor parameters:
  - `ksize`: 31
  - `sigma`: 4.0
  - `theta_values`: [0, π/4, π/2, 3π/4]
  - `lambda`: 10.0
  - `gamma`: 0.5

## API Endpoints

### POST /api/register
Register a new user with captured face images.

**Request Body:**
```json
{
  "user_id": "john_doe",
  "images": ["base64_image1", "base64_image2", "base64_image3"]
}
```

**Response:**
```json
{
  "success": true,
  "message": "User john_doe registered successfully with 3 face samples."
}
```

### POST /api/authenticate
Authenticate a user by matching face features.

**Request Body:**
```json
{
  "image": "base64_image"
}
```

**Response:**
```json
{
  "success": true,
  "user_id": "john_doe",
  "confidence": 0.87,
  "message": "Welcome back, john_doe!"
}
```

## Troubleshooting

### Model Not Found Error
If you see "Feature extractor not found" error:
1. Ensure you've trained the model using the Jupyter notebook
2. Run `python save_model.py` to create the feature extractor
3. Check that `feature_extractor.h5` exists in the project directory

### Camera Access Issues
If the webcam doesn't work:
1. Ensure your browser has permission to access the camera
2. Use HTTPS or localhost (required by modern browsers)
3. Check that no other application is using the camera

### Low Authentication Accuracy
To improve accuracy:
1. Capture more images during registration (5 recommended)
2. Ensure good lighting conditions
3. Look directly at the camera
4. Vary your pose slightly between captures
5. Adjust the `SIMILARITY_THRESHOLD` if needed

## Security Considerations

- User features are stored locally in `user_data.json`
- For production use, consider encrypting the feature database
- Implement rate limiting on authentication endpoints
- Add HTTPS in production environments
- Consider adding face liveness detection to prevent spoofing

## Technologies Used

- **Flask**: Web framework
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Image processing and face detection
- **NumPy**: Numerical computations
- **scikit-learn**: Cosine similarity calculation
- **JavaScript**: Frontend webcam access
- **HTML/CSS**: User interface

## License

This project is for educational purposes.

## Acknowledgments

- Gabor Transform for texture feature extraction
- VGG-style CNN architecture for feature learning
- Haar Cascade for face detection