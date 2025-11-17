from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import json
import os
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Global variables
feature_extractor = None
user_database_path = 'user_data.json'

# Gabor parameters (from notebook)
GABOR_PARAMS = {
    'ksize': 31,
    'sigma': 4.0,
    'theta_values': [0, np.pi/4, np.pi/2, 3*np.pi/4],
    'lambda': 10.0,
    'gamma': 0.5,
    'psi': 0
}

IMG_HEIGHT = 128
IMG_WIDTH = 128
SIMILARITY_THRESHOLD = 0.7

def load_feature_extractor():
    """Load the pre-trained feature extractor model"""
    global feature_extractor
    try:
        if os.path.exists('feature_extractor.h5'):
            feature_extractor = load_model('feature_extractor.h5')
            print("Feature extractor loaded successfully")
            return True
        else:
            print("Error: feature_extractor.h5 not found. Please run save_model.py first.")
            return False
    except Exception as e:
        print(f"Error loading feature extractor: {e}")
        return False

def create_gabor_kernels(params):
    """Create Gabor filter kernels"""
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
    """Apply Gabor transform to create 3-channel image"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    gabor_responses = []
    for kernel in kernels:
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        gabor_responses.append(filtered)
    
    gabor_features = np.array(gabor_responses)
    
    # Create 3 channels from Gabor responses
    channel_r = np.mean(gabor_features[:2], axis=0)
    channel_g = np.mean(gabor_features[2:], axis=0)
    channel_b = np.std(gabor_features, axis=0)
    
    # Normalize each channel
    channel_r = cv2.normalize(channel_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    channel_g = cv2.normalize(channel_g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    channel_b = cv2.normalize(channel_b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    gabor_image = np.stack([channel_r, channel_g, channel_b], axis=-1)
    return gabor_image

def preprocess_image(image):
    """Preprocess image: resize, apply Gabor transform, normalize"""
    gabor_kernels = create_gabor_kernels(GABOR_PARAMS)
    img = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    img = apply_gabor_transform(img, gabor_kernels)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def extract_features(image):
    """Extract 512-dim features from image using feature extractor"""
    if feature_extractor is None:
        raise Exception("Feature extractor not loaded")
    
    processed_img = preprocess_image(image)
    features = feature_extractor.predict(processed_img, verbose=0)
    return features.flatten()

def detect_face(image):
    """Detect face in image using Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return faces

def load_user_data():
    """Load user database from JSON file"""
    if os.path.exists(user_database_path):
        with open(user_database_path, 'r') as f:
            return json.load(f)
    return {}

def save_user_data(data):
    """Save user database to JSON file"""
    with open(user_database_path, 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/register')
def register():
    """Registration page"""
    return render_template('register.html')

@app.route('/authenticate')
def authenticate():
    """Authentication page"""
    return render_template('authenticate.html')

@app.route('/api/register', methods=['POST'])
def register_user():
    """Register a new user with captured face images"""
    if feature_extractor is None:
        return jsonify({
            'success': False,
            'message': 'Feature extractor not loaded. Please run save_model.py first.'
        })
    
    try:
        data = request.get_json()
        user_id = data.get('user_id', '').strip()
        images_data = data.get('images', [])
        
        if not user_id:
            return jsonify({'success': False, 'message': 'User ID is required'})
        
        if len(images_data) < 3:
            return jsonify({'success': False, 'message': 'Please capture at least 3 images'})
        
        # Process each image and extract features
        features_list = []
        for img_data in images_data:
            # Decode base64 image
            img_data = img_data.split(',')[1] if ',' in img_data else img_data
            img_bytes = base64.b64decode(img_data)
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            # Detect face
            faces = detect_face(img)
            if len(faces) == 0:
                continue
            
            # Extract the first detected face
            x, y, w, h = faces[0]
            face_img = img[y:y+h, x:x+w]
            
            # Extract features
            features = extract_features(face_img)
            features_list.append(features.tolist())
        
        if len(features_list) < 3:
            return jsonify({
                'success': False,
                'message': 'Could not detect faces in enough images. Please try again.'
            })
        
        # Average the features
        avg_features = np.mean(features_list, axis=0).tolist()
        
        # Load current user database
        user_data = load_user_data()
        
        # Check if user already exists
        if user_id in user_data:
            return jsonify({
                'success': False,
                'message': f'User {user_id} already exists. Please use a different ID.'
            })
        
        # Save user features
        user_data[user_id] = {
            'features': avg_features,
            'num_samples': len(features_list)
        }
        save_user_data(user_data)
        
        return jsonify({
            'success': True,
            'message': f'User {user_id} registered successfully with {len(features_list)} face samples.'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/authenticate', methods=['POST'])
def authenticate_user():
    """Authenticate user by matching face against database"""
    if feature_extractor is None:
        return jsonify({
            'success': False,
            'message': 'Feature extractor not loaded. Please run save_model.py first.'
        })
    
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image provided'})
        
        # Decode base64 image
        img_data = image_data.split(',')[1] if ',' in image_data else image_data
        img_bytes = base64.b64decode(img_data)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        # Detect face
        faces = detect_face(img)
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected. Please try again.'})
        
        # Extract face
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]
        
        # Extract features
        features = extract_features(face_img)
        
        # Load user database
        user_data = load_user_data()
        
        if not user_data:
            return jsonify({
                'success': False,
                'message': 'No registered users. Please register first.'
            })
        
        # Match against all users
        best_match_id = None
        best_similarity = 0.0
        
        for user_id, user_info in user_data.items():
            user_features = np.array(user_info['features'])
            similarity = cosine_similarity([features], [user_features])[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = user_id
        
        # Check if similarity exceeds threshold
        if best_similarity >= SIMILARITY_THRESHOLD:
            return jsonify({
                'success': True,
                'user_id': best_match_id,
                'confidence': float(best_similarity),
                'message': f'Welcome back, {best_match_id}!'
            })
        else:
            return jsonify({
                'success': False,
                'confidence': float(best_similarity),
                'message': f'Authentication failed. Confidence: {best_similarity:.2%} (threshold: {SIMILARITY_THRESHOLD:.2%})'
            })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

if __name__ == '__main__':
    # Load the feature extractor at startup
    if not load_feature_extractor():
        print("\n" + "="*70)
        print("WARNING: Feature extractor not found!")
        print("Please follow these steps:")
        print("1. Train the model using the Jupyter notebook")
        print("2. Run save_model.py to create the feature extractor")
        print("3. Run this app.py again")
        print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
