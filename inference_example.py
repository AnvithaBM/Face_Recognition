"""
Example script for using the trained hyperspectral face recognition model for inference.

This script demonstrates how to:
1. Load the trained model
2. Load the label encoder
3. Preprocess new images
4. Make predictions
5. Interpret results
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import pickle


class FaceRecognitionInference:
    """Class to handle face recognition inference using the trained model."""
    
    def __init__(self, model_path='hyperspectral_face_recognition_model.h5', 
                 encoder_path='label_encoder.pkl'):
        """
        Initialize the inference class.
        
        Args:
            model_path: Path to the trained model file
            encoder_path: Path to the label encoder file
        """
        # Load the trained model
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Load the label encoder
        print(f"Loading label encoder from {encoder_path}...")
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("Label encoder loaded successfully!")
        
        # Get input shape from model
        self.input_shape = self.model.input_shape[1:3]  # (height, width)
        print(f"Model expects input shape: {self.input_shape}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for model inference.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Preprocessed image array ready for prediction
        """
        try:
            # Load image
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                # Try with PIL as fallback
                img = np.array(Image.open(image_path))
            
            # Handle different image formats
            if len(img.shape) == 2:  # Grayscale
                img = np.stack([img] * 3, axis=-1)  # Convert to 3 channels
            elif img.shape[2] > 3:  # Hyperspectral with multiple bands
                # Select first 3 bands
                img = img[:, :, :3]
            
            # Resize to model's expected input size
            img = cv2.resize(img, self.input_shape)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
        
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def predict(self, image_path, top_k=3):
        """
        Make a prediction for a single image.
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess the image
        img = self.preprocess_image(image_path)
        if img is None:
            return None
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_batch, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_probs = predictions[top_indices]
        top_labels = self.label_encoder.inverse_transform(top_indices)
        
        # Prepare results
        results = {
            'predicted_label': top_labels[0],
            'confidence': float(top_probs[0]),
            'top_predictions': [
                {
                    'label': label,
                    'confidence': float(prob)
                }
                for label, prob in zip(top_labels, top_probs)
            ]
        }
        
        return results
    
    def predict_batch(self, image_paths):
        """
        Make predictions for multiple images.
        
        Args:
            image_paths: List of paths to image files
        
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        return results


def main():
    """Example usage of the FaceRecognitionInference class."""
    
    print("=" * 70)
    print("HYPERSPECTRAL FACE RECOGNITION - INFERENCE EXAMPLE")
    print("=" * 70)
    
    # Initialize the inference class
    try:
        recognizer = FaceRecognitionInference(
            model_path='hyperspectral_face_recognition_model.h5',
            encoder_path='label_encoder.pkl'
        )
    except Exception as e:
        print(f"\nError: Could not load model or encoder.")
        print(f"Make sure you have trained the model first by running the notebook.")
        print(f"Error details: {e}")
        return
    
    print("\n" + "=" * 70)
    print("EXAMPLE USAGE")
    print("=" * 70)
    
    # Example: Predict a single image
    example_image = "path/to/your/image.png"
    print(f"\nTo predict a single image:")
    print(f"  result = recognizer.predict('{example_image}')")
    print(f"  print(f\"Predicted: {{result['predicted_label']}} with confidence {{result['confidence']:.2%}}\")")
    
    print(f"\nTo predict multiple images:")
    print(f"  images = ['image1.png', 'image2.png', 'image3.png']")
    print(f"  results = recognizer.predict_batch(images)")
    print(f"  for img, res in zip(images, results):")
    print(f"      print(f\"{{img}}: {{res['predicted_label']}} ({{res['confidence']:.2%}})\")")
    
    print("\n" + "=" * 70)
    print("AUTHENTICATION EXAMPLE")
    print("=" * 70)
    print("""
# Simple authentication check
def authenticate_person(image_path, expected_person, threshold=0.9):
    result = recognizer.predict(image_path)
    
    if result['predicted_label'] == expected_person and result['confidence'] >= threshold:
        return True, result['confidence']
    else:
        return False, result['confidence']

# Usage
is_authenticated, confidence = authenticate_person('test_image.png', 'Person_01', threshold=0.9)
if is_authenticated:
    print(f"Authentication successful! Confidence: {confidence:.2%}")
else:
    print(f"Authentication failed. Confidence: {confidence:.2%}")
    """)
    
    print("\n" + "=" * 70)
    print("READY FOR INFERENCE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
