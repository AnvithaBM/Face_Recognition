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
        
        # Try loading with different methods based on file format
        full_model = None
        load_error = None
        
        # First, try standard load
        try:
            full_model = load_model(model_path)
        except (OSError, IOError) as e:
            if "file signature not found" in str(e):
                load_error = e
                print(f"\nWarning: Standard HDF5 loading failed: {e}")
                print("Attempting to load as Keras 3.x format...")
                
                # Try loading with compile=False for Keras 3.x models
                try:
                    full_model = load_model(model_path, compile=False)
                    print("Successfully loaded with compile=False")
                except Exception as e2:
                    print(f"Failed to load with compile=False: {e2}")
                    
                    # Try renaming and loading as .keras
                    keras_path = model_path.replace('.h5', '.keras')
                    if keras_path != model_path and not os.path.exists(keras_path):
                        print(f"\nAttempting to load as .keras format...")
                        try:
                            import shutil
                            shutil.copy(model_path, keras_path)
                            full_model = load_model(keras_path, compile=False)
                            print(f"Successfully loaded as .keras format!")
                            model_path = keras_path  # Update model path for reference
                        except Exception as e3:
                            print(f"Failed to load as .keras: {e3}")
            else:
                raise
        
        if full_model is None:
            raise Exception(f"Could not load model. Original error: {load_error}")
        
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
        
        print("\n" + "="*70)
        print("TROUBLESHOOTING TIPS:")
        print("="*70)
        print("\n1. Check if your model file is corrupted:")
        print("   - File size should be reasonable (> 1 MB)")
        print("   - Try re-training and saving the model")
        print("\n2. If you get 'file signature not found' error:")
        print("   - Your model might be in Keras 3.x format")
        print("   - Try renaming: best_model.h5 -> best_model.keras")
        print("   - Or re-save using: model.save('best_model.h5', save_format='h5')")
        print("\n3. Check your Keras/TensorFlow version:")
        print("   - Run: python -c \"import tensorflow as tf; print(tf.__version__)\"")
        print("   - Keras 3.x uses a different format than Keras 2.x")
        print("\n4. Alternative: Use the notebook checkpoint file")
        print("   - Look for 'best_face_recognition_model.keras' in your directory")
        print("   - Or check for checkpoint files in the notebook")
        print("="*70)
        
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
