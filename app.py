import os
import json
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

# Import your existing functions (assuming they're in the notebook)
# You'll need to copy the relevant functions here or import from the notebook

app = Flask(__name__)
CORS(app)

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
MODEL_PATH = 'best_model.keras'
FEATURES_FILE = 'user_features.json'

# Gabor parameters (same as in your notebook)
GABOR_PARAMS = {
    'ksize': 31,
    'sigma': 4.0,
    'theta_values': [0, np.pi/4, np.pi/2, 3*np.pi/4],
    'lambda': 10.0,
    'gamma': 0.5,
    'psi': 0
}

# Global variables
model = None
feature_extractor = None
gabor_kernels = None
user_features = {}

def create_gabor_kernels(params):
    """Create Gabor filter kernels (copied from your notebook)"""
    kernels = []
    for theta in params['theta_values']:
        kernel = cv2.getGaborKernel(
            (params['ksize'], params['ksize']),
            params['sigma'],
            theta,
            params['lambda'],
            params['gamma'],
            params['psi'],
            ktype=cv2.CV_32F
        )
        kernels.append(kernel)
    return kernels

def apply_gabor_transform(image, kernels):
    """Apply Gabor transform (copied from your notebook)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    gabor_responses = []
    for kernel in kernels:
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        gabor_responses.append(filtered)
    
    gabor_features = np.array(gabor_responses)
    
    channel_r = np.mean(gabor_features[:2], axis=0)
    channel_g = np.mean(gabor_features[2:], axis=0)
    channel_b = np.std(gabor_features, axis=0)
    
    channel_r = cv2.normalize(channel_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    channel_g = cv2.normalize(channel_g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    channel_b = cv2.normalize(channel_b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    gabor_image = np.stack([channel_r, channel_g, channel_b], axis=-1)
    
    return gabor_image

def preprocess_image(image):
    """Preprocess image for model input"""
    # Resize
    img = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    
    # Apply Gabor transform
    img = apply_gabor_transform(img, gabor_kernels)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def load_model_and_features():
    """Load the trained model and user features"""
    global model, feature_extractor, gabor_kernels, user_features
    
    # Create Gabor kernels
    gabor_kernels = create_gabor_kernels(GABOR_PARAMS)
    
    # Load model
    if os.path.exists(MODEL_PATH):
        model = models.load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        
        # Create feature extractor (output of second last dense layer)
        feature_extractor = models.Model(
            inputs=model.input,
            outputs=model.layers[-3].output  # Dense(256) layer
        )
        print("Feature extractor created")
    else:
        print(f"Warning: Model file {MODEL_PATH} not found")
    
    # Load user features
    if os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE, 'r') as f:
            user_features = json.load(f)
        print(f"Loaded features for {len(user_features)} users")
    else:
        print(f"Features file {FEATURES_FILE} not found, starting with empty database")

def extract_features(image):
    """Extract features from preprocessed image"""
    if feature_extractor is None:
        return None
    
    img = np.expand_dims(image, axis=0)
    features = feature_extractor.predict(img, verbose=0)
    return features.flatten()

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/authenticate')
def authenticate_page():
    return render_template('authenticate.html')

@app.route('/register', methods=['POST'])
def register_user():
    data = request.json
    username = data.get('username')
    images_data = data.get('images', [])
    
    if not username or not images_data:
        return jsonify({'success': False, 'message': 'Username and images required'})
    
    if username in user_features:
        return jsonify({'success': False, 'message': 'User already exists'})
    
    features_list = []
    for img_data in images_data:
        try:
            # Decode base64 image
            img_data = img_data.split(',')[1]  # Remove data:image/jpeg;base64,
            img_bytes = base64.b64decode(img_data)
            img = Image.open(BytesIO(img_bytes))
            img_array = np.array(img)
            
            # Preprocess and extract features
            processed_img = preprocess_image(img_array)
            features = extract_features(processed_img)
            
            if features is not None:
                features_list.append(features)
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
    
    if len(features_list) < 5:  # Require at least 5 good images
        return jsonify({'success': False, 'message': 'Need at least 5 valid face images'})
    
    # Average features across all images
    avg_features = np.mean(features_list, axis=0)
    
    # Save to database
    user_features[username] = avg_features.tolist()
    save_user_features()
    
    return jsonify({'success': True, 'message': f'User {username} registered successfully'})

@app.route('/authenticate', methods=['POST'])
def authenticate_user():
    data = request.json
    img_data = data.get('image')
    
    if not img_data:
        return jsonify({'success': False, 'message': 'Image required'})
    
    try:
        # Decode base64 image
        img_data = img_data.split(',')[1]  # Remove data:image/jpeg;base64,
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        img_array = np.array(img)
        
        # Preprocess and extract features
        processed_img = preprocess_image(img_array)
        features = extract_features(processed_img)
        
        if features is None:
            return jsonify({'success': False, 'message': 'Could not extract features'})
        
        # Compare with all users
        best_match = None
        best_similarity = -1
        threshold = 0.7  # Adjust threshold as needed
        
        for username, user_feat in user_features.items():
            user_feat = np.array(user_feat)
            similarity = cosine_similarity(features, user_feat)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = username
        
        if best_match and best_similarity > threshold:
            return jsonify({
                'success': True, 
                'message': f'Authenticated as {best_match}',
                'user': best_match,
                'confidence': float(best_similarity)
            })
        else:
            return jsonify({
                'success': False, 
                'message': 'Authentication failed - no match found',
                'confidence': float(best_similarity) if best_match else 0
            })
            
    except Exception as e:
        print(f"Error during authentication: {e}")
        return jsonify({'success': False, 'message': 'Authentication error'})

def save_user_features():
    """Save user features to file"""
    with open(FEATURES_FILE, 'w') as f:
        json.dump(user_features, f)

if __name__ == '__main__':
    load_model_and_features()
    app.run(debug=True, host='0.0.0.0', port=5000)