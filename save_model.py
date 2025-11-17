"""
Script to create and save a feature extractor from the trained model.
This extracts the 512-dim features from the Dense(512) layer.
"""
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

def create_feature_extractor(model_path='best_model.h5', output_path='feature_extractor.h5'):
    """
    Load the trained model and create a feature extractor that outputs 512-dim features.
    
    Args:
        model_path: Path to the trained model file (best_model.h5)
        output_path: Path to save the feature extractor model
    """
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("\nPlease train the model first using the Jupyter notebook:")
        print("1. Open 'face_recognition_hyperspectral (3).ipynb'")
        print("2. Run all cells to train the model")
        print("3. The model will be saved as 'best_model.h5' or 'best_face_recognition_model.keras'")
        print("4. If saved as .keras, rename it to 'best_model.h5'")
        print("5. Then run this script again")
        return False
    
    try:
        print(f"Loading model from {model_path}...")
        full_model = load_model(model_path)
        
        print("\nModel architecture:")
        full_model.summary()
        
        # Find the Dense(512) layer
        feature_layer = None
        for i, layer in enumerate(full_model.layers):
            if 'dense' in layer.name.lower() and hasattr(layer, 'units') and layer.units == 512:
                feature_layer = layer
                print(f"\nFound feature layer: {layer.name} at index {i}")
                break
        
        if feature_layer is None:
            print("\nError: Could not find Dense(512) layer in the model")
            print("Attempting to use the second-to-last layer instead...")
            feature_layer = full_model.layers[-2]
            print(f"Using layer: {feature_layer.name}")
        
        # Create feature extractor model
        feature_extractor = Model(
            inputs=full_model.input,
            outputs=feature_layer.output
        )
        
        print(f"\nFeature extractor created successfully!")
        print(f"Input shape: {feature_extractor.input_shape}")
        print(f"Output shape: {feature_extractor.output_shape}")
        
        # Save the feature extractor
        feature_extractor.save(output_path)
        print(f"\nFeature extractor saved to '{output_path}'")
        
        print("\n" + "="*70)
        print("SUCCESS! Feature extractor is ready.")
        print("You can now run the Flask app with: python app.py")
        print("="*70)
        
        return True
    
    except Exception as e:
        print(f"\nError creating feature extractor: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # Check for alternative model file names
    model_files = ['best_model.h5', 'best_face_recognition_model.keras', 'best_face_recognition_model.h5']
    model_path = None
    
    for file in model_files:
        if os.path.exists(file):
            model_path = file
            print(f"Found model file: {file}")
            break
    
    if model_path is None:
        print("No trained model found!")
        print("\nSearching for any .h5 or .keras files...")
        import glob
        model_files_found = glob.glob('*.h5') + glob.glob('*.keras')
        if model_files_found:
            print(f"Found these model files: {model_files_found}")
            model_path = model_files_found[0]
            print(f"Using: {model_path}")
        else:
            print("\nNo model files found in current directory.")
            print("Please train the model first using the Jupyter notebook.")
            sys.exit(1)
    
    # Create feature extractor
    success = create_feature_extractor(model_path)
    
    if not success:
        sys.exit(1)
