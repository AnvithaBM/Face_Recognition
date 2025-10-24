#!/usr/bin/env python3
"""
Quick demo script that creates a test model and demonstrates the authentication system.
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("=" * 70)
print("HYPERSPECTRAL FACE AUTHENTICATION SYSTEM - QUICK DEMO")
print("=" * 70)

# Step 1: Create a test model if it doesn't exist
if not os.path.exists('hyperspectral_embedding_model.h5'):
    print("\n[1/3] Creating test model...")
    
    @tf.keras.utils.register_keras_serializable()
    class L2Normalize(layers.Layer):
        def __init__(self, **kwargs):
            super(L2Normalize, self).__init__(**kwargs)
        
        def call(self, inputs):
            return tf.nn.l2_normalize(inputs, axis=1)
        
        def get_config(self):
            return super(L2Normalize, self).get_config()
    
    def build_test_model(input_shape=(128, 128, 33), embedding_dim=128):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        embeddings = layers.Dense(embedding_dim, activation=None)(x)
        embeddings_norm = L2Normalize()(embeddings)
        model = models.Model(inputs=inputs, outputs=embeddings_norm)
        return model
    
    test_model = build_test_model()
    test_model.save('hyperspectral_embedding_model.h5')
    print("  ✓ Test model created")
else:
    print("\n[1/3] Using existing model...")
    print("  ✓ Model found")

# Step 2: Import and initialize the authentication system
print("\n[2/3] Initializing authentication system...")
from face_authentication_system import HyperspectralFaceAuthenticator, create_synthetic_hyperspectral_image

authenticator = HyperspectralFaceAuthenticator(
    model_path='hyperspectral_embedding_model.h5',
    database_path='demo_database.pkl',
    similarity_metric='cosine',
    threshold=0.6
)
print("  ✓ System initialized")

# Step 3: Run demo
print("\n[3/3] Running authentication demo...")
print("\n" + "=" * 70)
print("DEMO: User Enrollment and Authentication")
print("=" * 70)

# Enroll users
print("\n➤ Enrolling users...")
alice_images = [create_synthetic_hyperspectral_image() for _ in range(3)]
bob_images = [create_synthetic_hyperspectral_image() for _ in range(3)]
charlie_images = [create_synthetic_hyperspectral_image() for _ in range(3)]

authenticator.enroll_user('alice', alice_images)
authenticator.enroll_user('bob', bob_images)
authenticator.enroll_user('charlie', charlie_images)

print(f"\n  Total users enrolled: {len(authenticator.list_users())}")
print(f"  Users: {authenticator.list_users()}")

# Test 1:1 Verification
print("\n" + "=" * 70)
print("TEST 1: 1:1 VERIFICATION (Specific User)")
print("=" * 70)

print("\n➤ Test A: Alice verifying as Alice (genuine)")
test_image = alice_images[0] + np.random.randn(*alice_images[0].shape) * 0.1
authenticated, matched_user, score = authenticator.authenticate(test_image, user_id='alice')
print(f"  Result: {'✓ AUTHENTICATED' if authenticated else '✗ REJECTED'}")
print(f"  Score: {score:.4f} (threshold: {authenticator.threshold})")

print("\n➤ Test B: Bob trying to verify as Alice (imposter)")
test_image = bob_images[0]
authenticated, matched_user, score = authenticator.authenticate(test_image, user_id='alice')
print(f"  Result: {'✓ REJECTED' if not authenticated else '✗ ACCEPTED'}")
print(f"  Score: {score:.4f} (threshold: {authenticator.threshold})")

# Test 1:N Identification
print("\n" + "=" * 70)
print("TEST 2: 1:N IDENTIFICATION (Find Best Match)")
print("=" * 70)

print("\n➤ Test C: Identifying Charlie from all users")
test_image = charlie_images[0] + np.random.randn(*charlie_images[0].shape) * 0.1
authenticated, matched_user, score = authenticator.authenticate(test_image, user_id=None)
print(f"  Result: {'✓ IDENTIFIED' if authenticated else '✗ NOT FOUND'}")
print(f"  Matched User: {matched_user}")
print(f"  Score: {score:.4f} (threshold: {authenticator.threshold})")

print("\n➤ Test D: Unknown person (not enrolled)")
unknown_image = create_synthetic_hyperspectral_image()
authenticated, matched_user, score = authenticator.authenticate(unknown_image, user_id=None)
print(f"  Result: {'✓ REJECTED' if not authenticated else '✗ ACCEPTED'}")
print(f"  Best Match: {matched_user}")
print(f"  Score: {score:.4f} (threshold: {authenticator.threshold})")

# Database management
print("\n" + "=" * 70)
print("TEST 3: DATABASE MANAGEMENT")
print("=" * 70)

print("\n➤ Current database:")
for user in authenticator.list_users():
    info = authenticator.get_user_info(user)
    print(f"  • {user}: {info['num_samples']} samples")

print("\n➤ Adding new user 'david'...")
david_images = [create_synthetic_hyperspectral_image() for _ in range(2)]
authenticator.enroll_user('david', david_images)

print(f"\n➤ Removing user 'david'...")
authenticator.remove_user('david')

print(f"\n➤ Final user list: {authenticator.list_users()}")

# Cleanup
print("\n" + "=" * 70)
print("DEMO COMPLETED")
print("=" * 70)
print("\nNote: This demo uses synthetic data and a simplified model.")
print("For production use:")
print("  1. Train the full model using 'hyperspectral_face_recognition_model.ipynb'")
print("  2. Use real hyperspectral face images")
print("  3. Tune the threshold based on your security requirements")

# Clean up demo files
if os.path.exists('demo_database.pkl'):
    os.remove('demo_database.pkl')
    print("\n✓ Demo database cleaned up")
