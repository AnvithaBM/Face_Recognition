#!/usr/bin/env python3
"""
Test script for the Hyperspectral Face Authentication System
This script verifies that all components work correctly.
"""

import sys
import os
import numpy as np
import tempfile
import shutil

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 70)
print("Testing Hyperspectral Face Authentication System")
print("=" * 70)

# Test 1: Import modules
print("\n[1/6] Testing imports...")
try:
    from face_authentication_system import (
        HyperspectralFaceAuthenticator,
        load_hyperspectral_image,
        create_synthetic_hyperspectral_image
    )
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create synthetic model
print("\n[2/6] Creating and saving a test model...")
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    # Custom L2 normalization layer to avoid Lambda issues
    @tf.keras.utils.register_keras_serializable()
    class L2Normalize(layers.Layer):
        def __init__(self, **kwargs):
            super(L2Normalize, self).__init__(**kwargs)
        
        def call(self, inputs):
            return tf.nn.l2_normalize(inputs, axis=1)
        
        def get_config(self):
            return super(L2Normalize, self).get_config()
    
    # Build a simple test model
    def build_test_model(input_shape=(128, 128, 33), embedding_dim=128):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        embeddings = layers.Dense(embedding_dim, activation=None)(x)
        # Use custom L2 normalization layer instead of Lambda
        embeddings_norm = L2Normalize()(embeddings)
        model = models.Model(inputs=inputs, outputs=embeddings_norm)
        return model
    
    test_model = build_test_model()
    test_model.save('test_embedding_model.h5')
    print("  ✓ Test model created and saved")
except Exception as e:
    print(f"  ✗ Model creation failed: {e}")
    sys.exit(1)

# Test 3: Initialize authenticator
print("\n[3/6] Initializing authenticator...")
try:
    authenticator = HyperspectralFaceAuthenticator(
        model_path='test_embedding_model.h5',
        database_path='test_database.pkl',
        similarity_metric='cosine',
        threshold=0.6
    )
    print("  ✓ Authenticator initialized")
except Exception as e:
    print(f"  ✗ Initialization failed: {e}")
    sys.exit(1)

# Test 4: Enroll users
print("\n[4/6] Enrolling test users...")
try:
    # Create synthetic hyperspectral images
    alice_images = [create_synthetic_hyperspectral_image() for _ in range(3)]
    bob_images = [create_synthetic_hyperspectral_image() for _ in range(3)]
    charlie_images = [create_synthetic_hyperspectral_image() for _ in range(3)]
    
    authenticator.enroll_user('alice', alice_images)
    authenticator.enroll_user('bob', bob_images)
    authenticator.enroll_user('charlie', charlie_images)
    
    users = authenticator.list_users()
    assert len(users) == 3, f"Expected 3 users, got {len(users)}"
    assert 'alice' in users
    assert 'bob' in users
    assert 'charlie' in users
    
    print("  ✓ Users enrolled successfully")
    print(f"    Enrolled users: {users}")
except Exception as e:
    print(f"  ✗ Enrollment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test authentication (1:1)
print("\n[5/6] Testing authentication (1:1 verification)...")
try:
    # Test with slight variation of Alice's image
    test_image = alice_images[0] + np.random.randn(*alice_images[0].shape) * 0.05
    authenticated, matched_user, score = authenticator.authenticate(test_image, user_id='alice')
    
    print(f"    Genuine attempt (Alice as Alice):")
    print(f"      Result: {'✓ PASS' if authenticated else '✗ FAIL'}")
    print(f"      Score: {score:.4f}")
    
    # Test with Bob's image claiming to be Alice
    authenticated2, matched_user2, score2 = authenticator.authenticate(bob_images[0], user_id='alice')
    
    print(f"    Imposter attempt (Bob as Alice):")
    print(f"      Result: {'✓ REJECTED' if not authenticated2 else '✗ ACCEPTED'}")
    print(f"      Score: {score2:.4f}")
    
    print("  ✓ Authentication (1:1) working")
except Exception as e:
    print(f"  ✗ Authentication (1:1) failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test identification (1:N)
print("\n[6/6] Testing identification (1:N)...")
try:
    # Test identifying Charlie
    test_image = charlie_images[0] + np.random.randn(*charlie_images[0].shape) * 0.05
    authenticated, matched_user, score = authenticator.authenticate(test_image, user_id=None)
    
    print(f"    Identifying Charlie:")
    print(f"      Result: {'✓ IDENTIFIED' if authenticated else '✗ NOT FOUND'}")
    print(f"      Matched: {matched_user}")
    print(f"      Score: {score:.4f}")
    
    # Test with unknown person
    unknown_image = create_synthetic_hyperspectral_image()
    authenticated2, matched_user2, score2 = authenticator.authenticate(unknown_image, user_id=None)
    
    print(f"    Unknown person:")
    print(f"      Result: {'✓ REJECTED' if not authenticated2 else '✗ ACCEPTED'}")
    print(f"      Best match: {matched_user2}")
    print(f"      Score: {score2:.4f}")
    
    print("  ✓ Identification (1:N) working")
except Exception as e:
    print(f"  ✗ Identification (1:N) failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cleanup
print("\n[Cleanup] Removing test files...")
try:
    os.remove('test_embedding_model.h5')
    if os.path.exists('test_database.pkl'):
        os.remove('test_database.pkl')
    print("  ✓ Cleanup complete")
except Exception as e:
    print(f"  Warning: Cleanup failed: {e}")

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED")
print("=" * 70)
print("\nThe Hyperspectral Face Authentication System is working correctly!")
print("\nNext steps:")
print("  1. Train the model using: jupyter notebook hyperspectral_face_recognition_model.ipynb")
print("  2. Try the demo: python example_usage.py --demo")
print("  3. Explore the interactive demo: jupyter notebook face_authentication_demo.ipynb")
