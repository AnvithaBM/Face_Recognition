"""
Feature Extractor Module for Face Authentication.
Loads the trained model and extracts face embeddings.
"""

import os
import numpy as np
import tensorflow as tf
from keras import models
import warnings
warnings.filterwarnings('ignore')

from utils import preprocess_image


class FeatureExtractor:
    """
    Feature extractor using a trained CNN model.
    Extracts face embeddings from the penultimate layer.
    """
    
    def __init__(self, model_path='best_model.h5', use_gabor=True):
        """
        Initialize the feature extractor.
        
        Args:
            model_path: Path to the trained model file
            use_gabor: Whether to use Gabor transform in preprocessing
        """
        self.model_path = model_path
        self.use_gabor = use_gabor
        self.model = None
        self.feature_model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and create feature extraction model."""
        if not os.path.exists(self.model_path):
            print(f"Warning: Model file '{self.model_path}' not found.")
            print("Creating a dummy model for demonstration...")
            self._create_dummy_model()
            return
        
        try:
            # Load the full model
            self.model = models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            
            # Create a feature extraction model (output from penultimate layer)
            # The penultimate layer is typically the last Dense layer before softmax
            feature_layer = None
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Dense) and layer != self.model.layers[-1]:
                    feature_layer = layer
                    break
            
            if feature_layer is None:
                # Fallback: use the layer before the last one
                feature_layer = self.model.layers[-2]
            
            self.feature_model = models.Model(
                inputs=self.model.input,
                outputs=feature_layer.output
            )
            print(f"Feature extraction model created using layer: {feature_layer.name}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Creating a dummy model for demonstration...")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """
        Create a dummy model for demonstration when actual model is not available.
        This allows the system to be tested without a pre-trained model.
        """
        from keras import layers
        
        # Determine input shape based on Gabor setting
        input_channels = 4 if self.use_gabor else 3
        input_shape = (128, 128, input_channels)
        
        # Create a simple CNN model
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),  # Feature layer
            layers.Dense(10, activation='softmax')  # Output layer
        ])
        
        # Build the model to establish input/output shapes
        model.build(input_shape=(None,) + input_shape)
        
        self.model = model
        # Use the second-to-last layer for features
        self.feature_model = models.Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[-2].output
        )
        
        print("Dummy model created successfully (256-dimensional features)")
        print("Note: This is for demonstration only. Load a trained model for actual use.")
    
    def extract_features(self, image_input):
        """
        Extract features from a single image.
        
        Args:
            image_input: Image as file path, PIL Image, or numpy array
        
        Returns:
            Feature embedding as numpy array
        """
        try:
            # Preprocess the image
            processed_img = preprocess_image(
                image_input,
                target_size=(128, 128),
                use_gabor=self.use_gabor
            )
            
            # Add batch dimension
            img_batch = np.expand_dims(processed_img, axis=0)
            
            # Extract features
            features = self.feature_model.predict(img_batch, verbose=0)
            
            # Return as 1D array
            return features.flatten()
        
        except Exception as e:
            raise ValueError(f"Error extracting features: {str(e)}")
    
    def extract_features_batch(self, image_inputs):
        """
        Extract features from multiple images.
        
        Args:
            image_inputs: List of images (file paths, PIL Images, or numpy arrays)
        
        Returns:
            List of feature embeddings
        """
        features_list = []
        
        for image_input in image_inputs:
            try:
                features = self.extract_features(image_input)
                features_list.append(features)
            except Exception as e:
                print(f"Warning: Failed to extract features from one image: {str(e)}")
                continue
        
        return features_list
    
    def get_feature_dimension(self):
        """
        Get the dimension of feature vectors.
        
        Returns:
            Integer representing feature dimension
        """
        if self.feature_model is None:
            return None
        
        return int(self.feature_model.output.shape[-1])
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        try:
            input_shape = self.model.input_shape if hasattr(self.model, 'input_shape') else "N/A"
        except:
            input_shape = "N/A"
        
        info = {
            "model_path": self.model_path,
            "use_gabor": self.use_gabor,
            "input_shape": input_shape,
            "feature_dimension": self.get_feature_dimension(),
            "total_parameters": self.model.count_params(),
        }
        
        return info


# Example usage
if __name__ == "__main__":
    # Test the feature extractor
    print("Testing FeatureExtractor...")
    
    # Initialize with dummy model (since we don't have a trained model yet)
    extractor = FeatureExtractor(model_path='nonexistent_model.h5', use_gabor=True)
    
    # Print model info
    info = extractor.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create a test image
    test_image = np.random.rand(128, 128, 3) * 255
    test_image = test_image.astype(np.uint8)
    
    # Extract features
    features = extractor.extract_features(test_image)
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Feature vector (first 10 values): {features[:10]}")
