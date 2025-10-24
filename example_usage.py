#!/usr/bin/env python3
"""
Example script demonstrating the Hyperspectral Face Authentication System

This script shows how to use the authentication system in a standalone application.
"""

import argparse
import numpy as np
from pathlib import Path
from face_authentication_system import (
    HyperspectralFaceAuthenticator,
    load_hyperspectral_image,
    create_synthetic_hyperspectral_image
)


def enroll_user(authenticator, user_id, image_paths):
    """Enroll a new user with provided images."""
    print(f"\nEnrolling user: {user_id}")
    print(f"Loading {len(image_paths)} images...")
    
    images = []
    for img_path in image_paths:
        if Path(img_path).exists():
            img = load_hyperspectral_image(img_path)
            images.append(img)
            print(f"  ✓ Loaded {img_path}")
        else:
            print(f"  ✗ File not found: {img_path}")
    
    if len(images) == 0:
        print("No valid images provided. Using synthetic data for demonstration.")
        images = [create_synthetic_hyperspectral_image() for _ in range(3)]
    
    success = authenticator.enroll_user(user_id, images)
    
    if success:
        print(f"\n✓ Successfully enrolled {user_id} with {len(images)} images")
    else:
        print(f"\n✗ Failed to enroll {user_id}")
    
    return success


def authenticate_user(authenticator, image_path, user_id=None):
    """Authenticate a user with the provided image."""
    print(f"\nAuthenticating...")
    
    if Path(image_path).exists():
        image = load_hyperspectral_image(image_path)
        print(f"  ✓ Loaded image: {image_path}")
    else:
        print(f"  ✗ File not found: {image_path}")
        print("  Using synthetic data for demonstration")
        image = create_synthetic_hyperspectral_image()
    
    if user_id:
        print(f"  Mode: 1:1 Verification (claiming to be '{user_id}')")
    else:
        print(f"  Mode: 1:N Identification (searching all users)")
    
    authenticated, matched_user, score = authenticator.authenticate(image, user_id=user_id)
    
    print("\n" + "=" * 60)
    if authenticated:
        print(f"✓ AUTHENTICATION SUCCESSFUL")
        print(f"  User: {matched_user}")
        print(f"  Confidence Score: {score:.4f}")
    else:
        print(f"✗ AUTHENTICATION FAILED")
        if matched_user:
            print(f"  Best Match: {matched_user} (score: {score:.4f})")
        else:
            print(f"  Score: {score:.4f}")
        print(f"  Threshold: {authenticator.threshold}")
    print("=" * 60)
    
    return authenticated


def list_users(authenticator):
    """List all enrolled users."""
    users = authenticator.list_users()
    
    if len(users) == 0:
        print("\nNo users enrolled in the database.")
        return
    
    print(f"\nEnrolled Users ({len(users)}):")
    print("=" * 60)
    
    for user in users:
        info = authenticator.get_user_info(user)
        print(f"  • {user}")
        print(f"    - Samples: {info['num_samples']}")
        print(f"    - Embedding Dimension: {info['embedding_dim']}")
    
    print("=" * 60)


def remove_user(authenticator, user_id):
    """Remove a user from the database."""
    print(f"\nRemoving user: {user_id}")
    success = authenticator.remove_user(user_id)
    
    if success:
        print(f"✓ User {user_id} removed successfully")
    
    return success


def run_demo(authenticator):
    """Run a complete demonstration of the system."""
    print("\n" + "=" * 70)
    print(" HYPERSPECTRAL FACE AUTHENTICATION SYSTEM - DEMO")
    print("=" * 70)
    
    # Demo: Enroll synthetic users
    print("\n1. ENROLLING DEMO USERS")
    print("-" * 70)
    
    demo_users = ['alice', 'bob', 'charlie']
    user_images = {}
    
    for user in demo_users:
        images = [create_synthetic_hyperspectral_image() for _ in range(3)]
        user_images[user] = images
        authenticator.enroll_user(user, images)
    
    # Demo: List users
    print("\n2. LISTING ENROLLED USERS")
    print("-" * 70)
    list_users(authenticator)
    
    # Demo: 1:1 Verification
    print("\n3. TESTING 1:1 VERIFICATION")
    print("-" * 70)
    
    # Genuine attempt
    print("\nTest A: Alice authenticating as Alice (genuine)")
    test_image = user_images['alice'][0] + np.random.randn(*user_images['alice'][0].shape) * 0.1
    authenticated, matched_user, score = authenticator.authenticate(test_image, user_id='alice')
    status = "✓ SUCCESS" if authenticated else "✗ FAILED"
    print(f"  Result: {status} (score: {score:.4f})")
    
    # Imposter attempt
    print("\nTest B: Bob trying to authenticate as Alice (imposter)")
    test_image = user_images['bob'][0]
    authenticated, matched_user, score = authenticator.authenticate(test_image, user_id='alice')
    status = "✓ REJECTED" if not authenticated else "✗ ACCEPTED"
    print(f"  Result: {status} (score: {score:.4f})")
    
    # Demo: 1:N Identification
    print("\n4. TESTING 1:N IDENTIFICATION")
    print("-" * 70)
    
    print("\nTest C: Identifying Charlie from all users")
    test_image = user_images['charlie'][0] + np.random.randn(*user_images['charlie'][0].shape) * 0.1
    authenticated, matched_user, score = authenticator.authenticate(test_image, user_id=None)
    if authenticated and matched_user == 'charlie':
        print(f"  Result: ✓ CORRECTLY IDENTIFIED as {matched_user} (score: {score:.4f})")
    else:
        print(f"  Result: ✗ MISIDENTIFIED as {matched_user} (score: {score:.4f})")
    
    # Demo: Unknown person
    print("\nTest D: Attempting with unknown person")
    unknown_image = create_synthetic_hyperspectral_image()
    authenticated, matched_user, score = authenticator.authenticate(unknown_image, user_id=None)
    status = "✓ CORRECTLY REJECTED" if not authenticated else "✗ INCORRECTLY ACCEPTED"
    print(f"  Result: {status} (score: {score:.4f})")
    
    # Demo: Threshold testing
    print("\n5. TESTING DIFFERENT THRESHOLDS")
    print("-" * 70)
    
    test_image = user_images['bob'][0] + np.random.randn(*user_images['bob'][0].shape) * 0.15
    
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        authenticator.update_threshold(threshold)
        authenticated, matched_user, score = authenticator.authenticate(test_image, user_id='bob')
        status = "PASS" if authenticated else "FAIL"
        print(f"  Threshold {threshold:.1f}: {status} (score: {score:.4f})")
    
    # Reset threshold
    authenticator.update_threshold(0.6)
    
    print("\n" + "=" * 70)
    print(" DEMO COMPLETED")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Hyperspectral Face Authentication System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with synthetic data
  python example_usage.py --demo
  
  # Enroll a new user
  python example_usage.py --enroll alice --images img1.npy img2.npy img3.npy
  
  # Authenticate (1:1 verification)
  python example_usage.py --authenticate test.npy --user alice
  
  # Identify (1:N identification)
  python example_usage.py --authenticate test.npy
  
  # List all users
  python example_usage.py --list
  
  # Remove a user
  python example_usage.py --remove alice
        """
    )
    
    parser.add_argument('--model', type=str, default='hyperspectral_embedding_model.h5',
                       help='Path to the embedding model (default: hyperspectral_embedding_model.h5)')
    parser.add_argument('--database', type=str, default='user_database.pkl',
                       help='Path to the user database (default: user_database.pkl)')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'],
                       help='Similarity metric (default: cosine)')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Authentication threshold (default: 0.6)')
    
    # Actions
    parser.add_argument('--demo', action='store_true',
                       help='Run a complete demonstration')
    parser.add_argument('--enroll', type=str, metavar='USER_ID',
                       help='Enroll a new user')
    parser.add_argument('--images', nargs='+', metavar='IMAGE',
                       help='Hyperspectral image files for enrollment')
    parser.add_argument('--authenticate', type=str, metavar='IMAGE',
                       help='Authenticate using this image')
    parser.add_argument('--user', type=str, metavar='USER_ID',
                       help='User ID for 1:1 verification (optional)')
    parser.add_argument('--list', action='store_true',
                       help='List all enrolled users')
    parser.add_argument('--remove', type=str, metavar='USER_ID',
                       help='Remove a user from the database')
    
    args = parser.parse_args()
    
    # Initialize authenticator
    print("Initializing Hyperspectral Face Authentication System...")
    print(f"  Model: {args.model}")
    print(f"  Database: {args.database}")
    print(f"  Metric: {args.metric}")
    print(f"  Threshold: {args.threshold}")
    
    try:
        authenticator = HyperspectralFaceAuthenticator(
            model_path=args.model,
            database_path=args.database,
            similarity_metric=args.metric,
            threshold=args.threshold
        )
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure you have:")
        print("  1. Trained the model using 'hyperspectral_face_recognition_model.ipynb'")
        print("  2. The model file exists at the specified path")
        return
    
    # Execute requested action
    if args.demo:
        run_demo(authenticator)
    
    elif args.enroll:
        if not args.images:
            print("✗ Error: --images required for enrollment")
            return
        enroll_user(authenticator, args.enroll, args.images)
    
    elif args.authenticate:
        authenticate_user(authenticator, args.authenticate, args.user)
    
    elif args.list:
        list_users(authenticator)
    
    elif args.remove:
        remove_user(authenticator, args.remove)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
