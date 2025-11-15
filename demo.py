#!/usr/bin/env python3
"""
Demo script to test the Face Authentication System.
Creates synthetic test data and demonstrates all features.
"""

import numpy as np
from PIL import Image
import os

from face_authentication import FaceAuthenticationSystem


def create_synthetic_face(base_pattern, variation=0.1):
    """Create a synthetic face image with some variation."""
    noise = np.random.normal(0, variation, base_pattern.shape)
    img = np.clip(base_pattern + noise, 0, 255)
    return img.astype(np.uint8)


def main():
    print("=" * 70)
    print("Face Authentication System - Demo")
    print("=" * 70)
    
    # Initialize system
    print("\n[1] Initializing authentication system...")
    auth_system = FaceAuthenticationSystem(
        model_path='best_model.h5',  # Will use dummy if not found
        database_path='demo_database.pkl',
        similarity_threshold=0.6,
        use_gabor=True
    )
    
    # Show statistics
    stats = auth_system.get_statistics()
    print("\nSystem Configuration:")
    print(f"  - Feature Dimension: {stats['feature_dimension']}")
    print(f"  - Similarity Threshold: {stats['similarity_threshold']}")
    print(f"  - Gabor Transform: {stats['use_gabor']}")
    print(f"  - Registered Users: {stats['total_users']}")
    
    # Create synthetic test users
    print("\n[2] Creating synthetic face images for testing...")
    
    # User 1: Alice
    alice_base = np.random.rand(128, 128, 3) * 255
    alice_samples = [create_synthetic_face(alice_base, 0.05) for _ in range(3)]
    
    # User 2: Bob
    bob_base = np.random.rand(128, 128, 3) * 255
    bob_samples = [create_synthetic_face(bob_base, 0.05) for _ in range(3)]
    
    # User 3: Charlie
    charlie_base = np.random.rand(128, 128, 3) * 255
    charlie_samples = [create_synthetic_face(charlie_base, 0.05) for _ in range(3)]
    
    print("  ✓ Created 3 users with 3 samples each")
    
    # Register users
    print("\n[3] Registering users...")
    
    users_to_register = [
        ('alice', alice_samples, {'name': 'Alice Johnson', 'department': 'Engineering'}),
        ('bob', bob_samples, {'name': 'Bob Smith', 'department': 'Marketing'}),
        ('charlie', charlie_samples, {'name': 'Charlie Brown', 'department': 'Sales'}),
    ]
    
    for user_id, samples, metadata in users_to_register:
        success, message = auth_system.register_user(user_id, samples, metadata)
        if success:
            print(f"  ✓ {message}")
        else:
            print(f"  ✗ {message}")
    
    # List registered users
    print("\n[4] Listing registered users...")
    users = auth_system.list_users()
    for user in users:
        print(f"  - {user['user_id']}: {user['num_samples']} samples")
        if user.get('metadata'):
            print(f"    Name: {user['metadata'].get('name', 'N/A')}")
    
    # Test authentication (1:N)
    print("\n[5] Testing Authentication (1:N Identification)...")
    
    # Test with Alice's face (should match)
    alice_test = create_synthetic_face(alice_base, 0.08)
    user_id, confidence, message = auth_system.authenticate_user(alice_test)
    print(f"\n  Test 1 - Alice's face:")
    print(f"    {message}")
    print(f"    Identified as: {user_id}")
    print(f"    Confidence: {confidence*100:.1f}%")
    
    # Test with Bob's face (should match)
    bob_test = create_synthetic_face(bob_base, 0.08)
    user_id, confidence, message = auth_system.authenticate_user(bob_test)
    print(f"\n  Test 2 - Bob's face:")
    print(f"    {message}")
    print(f"    Identified as: {user_id}")
    print(f"    Confidence: {confidence*100:.1f}%")
    
    # Test with unknown face (should fail)
    unknown_base = np.random.rand(128, 128, 3) * 255
    unknown_test = unknown_base.astype(np.uint8)
    user_id, confidence, message = auth_system.authenticate_user(unknown_test)
    print(f"\n  Test 3 - Unknown face:")
    print(f"    {message}")
    print(f"    Identified as: {user_id}")
    print(f"    Best match: {confidence*100:.1f}%")
    
    # Test verification (1:1)
    print("\n[6] Testing Verification (1:1 Matching)...")
    
    # Verify Alice with Alice's face (should succeed)
    verified, similarity, message = auth_system.verify_user('alice', alice_test)
    print(f"\n  Test 1 - Verify Alice with Alice's face:")
    print(f"    {message}")
    print(f"    Result: {'✓ VERIFIED' if verified else '✗ REJECTED'}")
    print(f"    Similarity: {similarity*100:.1f}%")
    
    # Verify Alice with Bob's face (should fail)
    verified, similarity, message = auth_system.verify_user('alice', bob_test)
    print(f"\n  Test 2 - Verify Alice with Bob's face:")
    print(f"    {message}")
    print(f"    Result: {'✓ VERIFIED' if verified else '✗ REJECTED'}")
    print(f"    Similarity: {similarity*100:.1f}%")
    
    # Test database operations
    print("\n[7] Testing database operations...")
    
    # Update user
    alice_new_samples = [create_synthetic_face(alice_base, 0.06) for _ in range(2)]
    success, message = auth_system.update_user('alice', alice_new_samples)
    print(f"  Update Alice: {message}")
    
    # Export database
    export_file = 'demo_database_export.json'
    auth_system.export_database(export_file, format='json')
    print(f"  ✓ Database exported to {export_file}")
    
    # Final statistics
    print("\n[8] Final statistics...")
    stats = auth_system.get_statistics()
    print(f"  - Total registered users: {stats['total_users']}")
    print(f"  - Feature dimension: {stats['feature_dimension']}")
    print(f"  - Threshold: {stats['similarity_threshold']}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    
    # Cleanup option
    print("\nNote: Demo database files created:")
    print("  - demo_database.pkl")
    print("  - demo_database_export.json")
    print("\nYou can delete these files if not needed.")


if __name__ == "__main__":
    main()
