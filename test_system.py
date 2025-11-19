#!/usr/bin/env python3
"""
Demo script to test the face authentication system components

This script tests the core functionality without requiring a web browser.
It's useful for:
- Verifying the system is set up correctly
- Testing Gabor filter transformations
- Checking model loading (if available)
- Validating feature extraction

Note: This doesn't test the full web interface, just the backend components.
"""

import os
import sys
import numpy as np
import cv2

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import flask
        print("  ✓ Flask imported")
    except ImportError as e:
        print(f"  ✗ Flask import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow imported (version {tf.__version__})")
    except ImportError as e:
        print(f"  ✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"  ✓ OpenCV imported (version {cv2.__version__})")
    except ImportError as e:
        print(f"  ✗ OpenCV import failed: {e}")
        return False
    
    try:
        import PIL
        print(f"  ✓ Pillow imported")
    except ImportError as e:
        print(f"  ✗ Pillow import failed: {e}")
        return False
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        print("  ✓ scikit-learn imported")
    except ImportError as e:
        print(f"  ✗ scikit-learn import failed: {e}")
        return False
    
    print("All imports successful!\n")
    return True

def test_gabor_filters():
    """Test Gabor filter creation and application"""
    print("Testing Gabor filters...")
    try:
        from app import create_gabor_kernels, apply_gabor_transform, GABOR_PARAMS
        
        # Create kernels
        kernels = create_gabor_kernels(GABOR_PARAMS)
        print(f"  ✓ Created {len(kernels)} Gabor kernels")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Apply Gabor transform
        transformed = apply_gabor_transform(test_image, kernels)
        print(f"  ✓ Applied Gabor transform")
        print(f"    Input shape: {test_image.shape}")
        print(f"    Output shape: {transformed.shape}")
        
        # Check output is valid
        assert transformed.shape == (128, 128, 3), "Output shape mismatch"
        assert transformed.dtype == np.uint8, "Output dtype mismatch"
        print("  ✓ Gabor transform output is valid\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Gabor filter test failed: {e}\n")
        return False

def test_model_loading():
    """Test model loading (if model file exists)"""
    print("Testing model loading...")
    try:
        from app import MODEL_PATH, load_model_and_features
        
        if os.path.exists(MODEL_PATH):
            print(f"  ✓ Model file found: {MODEL_PATH}")
            load_model_and_features()
            print("  ✓ Model loaded successfully")
            
            from app import model, feature_extractor
            if model is not None:
                print(f"    Model input shape: {model.input_shape}")
                print(f"    Model output shape: {model.output_shape}")
            if feature_extractor is not None:
                print(f"    Feature extractor output shape: {feature_extractor.output_shape}")
        else:
            print(f"  ⚠ Model file not found: {MODEL_PATH}")
            print("    This is expected if you haven't trained the model yet")
            print("    See MODEL_INFO.md for instructions on creating the model")
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Model loading test failed: {e}\n")
        return False

def test_preprocessing():
    """Test image preprocessing pipeline"""
    print("Testing image preprocessing...")
    try:
        from app import preprocess_image, IMG_WIDTH, IMG_HEIGHT, create_gabor_kernels, GABOR_PARAMS
        import app as app_module
        
        # Ensure gabor_kernels is initialized
        if app_module.gabor_kernels is None:
            app_module.gabor_kernels = create_gabor_kernels(GABOR_PARAMS)
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        # Preprocess
        processed = preprocess_image(test_image)
        
        print(f"  ✓ Preprocessing completed")
        print(f"    Input shape: {test_image.shape}")
        print(f"    Output shape: {processed.shape}")
        print(f"    Expected shape: ({IMG_HEIGHT}, {IMG_WIDTH}, 3)")
        print(f"    Value range: [{processed.min():.3f}, {processed.max():.3f}]")
        
        # Validate
        assert processed.shape == (IMG_HEIGHT, IMG_WIDTH, 3), "Shape mismatch"
        assert 0 <= processed.min() and processed.max() <= 1, "Value range incorrect"
        
        print("  ✓ Preprocessing output is valid\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Preprocessing test failed: {e}\n")
        return False

def test_feature_extraction():
    """Test feature extraction (if model is available)"""
    print("Testing feature extraction...")
    try:
        from app import extract_features, feature_extractor, preprocess_image
        
        if feature_extractor is None:
            print("  ⚠ Feature extractor not available (model not loaded)")
            print("    This test requires a trained model\n")
            return True
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        # Preprocess and extract features
        processed = preprocess_image(test_image)
        features = extract_features(processed)
        
        print(f"  ✓ Feature extraction completed")
        print(f"    Feature vector shape: {features.shape}")
        print(f"    Feature vector size: {len(features)}")
        
        # Validate
        assert len(features) > 0, "Empty feature vector"
        
        print("  ✓ Feature extraction successful\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Feature extraction test failed: {e}\n")
        return False

def test_similarity_calculation():
    """Test similarity calculation between feature vectors"""
    print("Testing similarity calculation...")
    try:
        from app import cosine_similarity
        
        # Create test vectors
        vec1 = np.random.randn(256)
        vec2 = vec1 + np.random.randn(256) * 0.1  # Similar vector
        vec3 = np.random.randn(256)  # Different vector
        
        # Calculate similarities
        sim_similar = cosine_similarity(vec1, vec2)
        sim_different = cosine_similarity(vec1, vec3)
        
        print(f"  ✓ Similarity calculation completed")
        print(f"    Similar vectors: {sim_similar:.4f}")
        print(f"    Different vectors: {sim_different:.4f}")
        
        # Validate
        assert -1 <= sim_similar <= 1, "Similarity out of range"
        assert -1 <= sim_different <= 1, "Similarity out of range"
        assert sim_similar > sim_different, "Similar vectors should have higher similarity"
        
        print("  ✓ Similarity calculation is correct\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Similarity calculation test failed: {e}\n")
        return False

def test_flask_app():
    """Test if Flask app can be created"""
    print("Testing Flask app...")
    try:
        from app import app
        
        print(f"  ✓ Flask app created")
        print(f"    App name: {app.name}")
        print(f"    Debug mode: {app.debug}")
        
        # List routes
        print("    Routes:")
        for rule in app.url_map.iter_rules():
            print(f"      - {rule.endpoint}: {rule.rule} {list(rule.methods)}")
        
        print("  ✓ Flask app is valid\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Flask app test failed: {e}\n")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Face Authentication System - Component Test")
    print("=" * 60)
    print()
    
    results = {
        "Imports": test_imports(),
        "Gabor Filters": test_gabor_filters(),
        "Model Loading": test_model_loading(),
        "Preprocessing": test_preprocessing(),
        "Feature Extraction": test_feature_extraction(),
        "Similarity Calculation": test_similarity_calculation(),
        "Flask App": test_flask_app(),
    }
    
    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print()
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    
    if failed == 0:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. If you don't have a model, train one using the Jupyter notebook")
        print("2. Start the Flask app: python app.py")
        print("3. Open browser to: http://localhost:5000")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
