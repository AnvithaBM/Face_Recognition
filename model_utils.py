import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from pathlib import Path

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

def create_gabor_kernels(params):
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
    gabor_kernels = create_gabor_kernels(GABOR_PARAMS)
    img = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    img = apply_gabor_transform(img, gabor_kernels)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def load_trained_model(model_path='best_model.h5'):
    if os.path.exists(model_path):
        model = load_model(model_path)
        # Remove the classification layer to get features
        feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
        return feature_extractor
    else:
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")

def extract_features(model, image):
    processed_img = preprocess_image(image)
    features = model.predict(processed_img, verbose=0)
    return features.flatten()

def load_user_data(data_path='users.json'):
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            return json.load(f)
    return {}

def save_user_data(data, data_path='users.json'):
    with open(data_path, 'w') as f:
        json.dump(data, f)

def register_user(user_id, face_images, model):
    features_list = []
    for img in face_images:
        features = extract_features(model, img)
        features_list.append(features.tolist())
    
    user_data = load_user_data()
    user_data[user_id] = features_list
    save_user_data(user_data)
    return True

def authenticate_user(face_image, model, threshold=0.8):
    features = extract_features(model, face_image)
    
    user_data = load_user_data()
    if not user_data:
        return None, 0.0
    
    best_match = None
    best_similarity = 0.0
    
    for user_id, user_features in user_data.items():
        for user_feat in user_features:
            similarity = cosine_similarity([features], [user_feat])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = user_id
    
    if best_similarity >= threshold:
        return best_match, best_similarity
    else:
        return None, best_similarity

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces
