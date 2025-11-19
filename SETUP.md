# Face Authentication System - Setup Guide

## Prerequisites

1. **Python 3.8 or higher** - Check your version:
   ```bash
   python3 --version
   ```

2. **pip** (Python package manager)

3. **Webcam** - Required for capturing face images

4. **Modern web browser** with camera access support (Chrome, Firefox, Safari, Edge)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/AnvithaBM/Face_Recognition.git
cd Face_Recognition
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install Flask Flask-CORS tensorflow opencv-python numpy Pillow scikit-learn
```

### 3. Train or Obtain the Model

The system requires a trained model file named `best_model.keras`. You have two options:

**Option A: Train Your Own Model**

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook "face_recognition_hyperspectral (3).ipynb"
   ```

2. Follow the notebook to train the model on your dataset

3. The notebook will save the model as `best_model.keras`

**Option B: Use Pre-trained Model**

If you have a pre-trained `best_model.keras` file, place it in the project root directory.

**Note**: The system will still run without the model file, but registration and authentication features will not work until a model is provided.

## Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. You should see output like:
   ```
   * Running on http://0.0.0.0:5000
   * Running on http://127.0.0.1:5000
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Using the System

### Registering a New User

1. Click "Register" on the home page
2. Enter a unique username
3. Click "Start Camera" to enable your webcam
4. Position your face in the camera frame
5. Click "Capture Image" multiple times (5-10 images recommended)
   - Try different angles: front, left, right
   - Maintain good lighting
   - Keep face clearly visible
6. Review captured images (you can remove any unwanted ones)
7. Click "Register User" to complete registration
8. Wait for success message

### Authenticating a User

1. Click "Authenticate" on the home page
2. Click "Start Camera" to enable your webcam
3. Position your face in the camera frame
4. Click "Authenticate"
5. The system will:
   - Extract features from your face
   - Compare with registered users
   - Display authentication result with confidence score

## Troubleshooting

### Camera Not Working

**Issue**: Camera doesn't start or shows error

**Solutions**:
- Grant camera permissions to your browser
- Check if another application is using the camera
- Try a different browser
- Ensure camera drivers are installed

### Model Not Found Warning

**Issue**: "Model file best_model.keras not found"

**Solution**: 
- Train the model using the provided notebook, OR
- Obtain a pre-trained model file and place it in the project root

### Import Errors

**Issue**: Module not found errors

**Solution**:
```bash
pip install --upgrade -r requirements.txt
```

### Port Already in Use

**Issue**: Port 5000 is already in use

**Solution**: Kill the process using port 5000:
```bash
# On Linux/Mac:
lsof -ti:5000 | xargs kill -9

# Or change the port in app.py (last line):
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Low Authentication Accuracy

**Possible Causes**:
- Poor lighting during registration or authentication
- Face not clearly visible
- Different camera angles
- Insufficient training images during registration

**Solutions**:
- Ensure good lighting conditions
- Capture more images during registration (8-10 recommended)
- Capture images from multiple angles
- Re-register the user with better quality images

## Security Considerations

⚠️ **Important Security Notes**:

1. **User Data**: Currently stored in `user_features.json` in plain text
   - For production: Use encrypted database
   - Implement proper access controls

2. **HTTPS**: Run over HTTPS in production to protect data transmission

3. **Liveness Detection**: Consider adding to prevent photo-based spoofing

4. **Authentication Threshold**: Current threshold is 0.7 (70% confidence)
   - Adjust in `app.py` if needed
   - Higher threshold = more strict (fewer false positives)
   - Lower threshold = more lenient (fewer false negatives)

## System Architecture

```
┌─────────────┐     HTTP      ┌──────────────┐
│   Browser   │ ←───────────→ │ Flask Server │
│ (HTML/JS)   │               │   (app.py)   │
└─────────────┘               └──────┬───────┘
                                     │
                              ┌──────┴───────┐
                              │              │
                        ┌─────▼─────┐  ┌────▼──────┐
                        │  VGG Model│  │   Gabor   │
                        │  (.keras) │  │  Filters  │
                        └───────────┘  └───────────┘
                              │
                        ┌─────▼──────┐
                        │  Features  │
                        │ (user_data)│
                        └────────────┘
```

## File Structure

```
Face_Recognition/
├── app.py                    # Main Flask application
├── model_utils.py            # Utility functions for model operations
├── face_auth.py              # Alternative authentication module
├── best_model.keras          # Trained model (to be added)
├── user_features.json        # Stored user embeddings (created at runtime)
├── requirements.txt          # Python dependencies
├── templates/               # HTML templates
│   ├── index.html
│   ├── register.html
│   └── authenticate.html
├── static/                  # Static assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── register.js
│       └── authenticate.js
└── face_recognition_hyperspectral (3).ipynb  # Model training notebook
```

## Performance Tips

1. **Image Quality**: 
   - Use good lighting
   - Keep face centered
   - Maintain consistent distance from camera

2. **Registration**:
   - Capture 8-10 images per user
   - Include different facial expressions
   - Vary head angles slightly

3. **Authentication Speed**:
   - First authentication may be slower (model loading)
   - Subsequent authentications are faster

## Next Steps

1. Train or obtain the `best_model.keras` file
2. Register test users
3. Test authentication with different lighting/angles
4. Adjust threshold if needed for your use case
5. Consider deploying to a server for production use

## Support

For issues or questions:
- Check the Troubleshooting section above
- Review the code comments in `app.py`
- Consult the Jupyter notebook for model details

## License

This is an educational project. See LICENSE file for details.
