from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from model_utils import load_trained_model, register_user, authenticate_user, detect_faces
import os

app = Flask(__name__)

# Load the model
try:
    model = load_trained_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/authenticate')
def authenticate():
    return render_template('authenticate.html')

@app.route('/register_user', methods=['POST'])
def register_user_endpoint():
    if not model:
        return jsonify({'success': False, 'message': 'Model not loaded'})
    
    data = request.get_json()
    user_id = data.get('user_id')
    images_data = data.get('images', [])
    
    if not user_id or not images_data:
        return jsonify({'success': False, 'message': 'Invalid data'})
    
    try:
        face_images = []
        for img_data in images_data:
            # Decode base64 image
            img_data = img_data.split(',')[1] if ',' in img_data else img_data
            img_bytes = base64.b64decode(img_data)
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            # Detect face
            faces = detect_faces(img)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = img[y:y+h, x:x+w]
                face_images.append(face)
        
        if len(face_images) < 3:
            return jsonify({'success': False, 'message': 'Not enough face images captured'})
        
        success = register_user(user_id, face_images, model)
        if success:
            return jsonify({'success': True, 'message': f'User {user_id} registered successfully'})
        else:
            return jsonify({'success': False, 'message': 'Registration failed'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/authenticate_user', methods=['POST'])
def authenticate_user_endpoint():
    if not model:
        return jsonify({'success': False, 'message': 'Model not loaded'})
    
    data = request.get_json()
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    try:
        # Decode base64 image
        img_data = image_data.split(',')[1] if ',' in image_data else image_data
        img_bytes = base64.b64decode(img_data)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        # Detect face
        faces = detect_faces(img)
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        
        user_id, similarity = authenticate_user(face, model)
        
        if user_id:
            return jsonify({
                'success': True, 
                'user_id': user_id, 
                'similarity': float(similarity),
                'message': f'Authenticated as {user_id}'
            })
        else:
            return jsonify({
                'success': False, 
                'similarity': float(similarity),
                'message': 'Authentication failed'
            })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)