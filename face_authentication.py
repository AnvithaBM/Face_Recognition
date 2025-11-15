"""
Face Authentication System.
Handles user registration and verification using face embeddings.
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from feature_extractor import FeatureExtractor
from utils import calculate_cosine_similarity, average_embeddings


class FaceAuthenticationSystem:
    """
    Complete face authentication system with registration and verification.
    """
    
    def __init__(
        self,
        model_path='best_model.h5',
        database_path='face_database.pkl',
        use_gabor=True,
        similarity_threshold=0.6
    ):
        """
        Initialize the authentication system.
        
        Args:
            model_path: Path to the trained model
            database_path: Path to store user database
            use_gabor: Whether to use Gabor transform
            similarity_threshold: Threshold for authentication (0-1)
        """
        self.model_path = model_path
        self.database_path = database_path
        self.use_gabor = use_gabor
        self.similarity_threshold = similarity_threshold
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            model_path=model_path,
            use_gabor=use_gabor
        )
        
        # Load or create database
        self.database = self._load_database()
    
    def _load_database(self) -> Dict:
        """
        Load the user database from disk.
        
        Returns:
            Dictionary containing user data
        """
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    database = pickle.load(f)
                print(f"Database loaded: {len(database)} users")
                return database
            except Exception as e:
                print(f"Error loading database: {str(e)}")
                print("Creating new database...")
        
        return {}
    
    def _save_database(self):
        """Save the user database to disk."""
        try:
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.database, f)
            print(f"Database saved successfully")
        except Exception as e:
            print(f"Error saving database: {str(e)}")
    
    def register_user(
        self,
        user_id: str,
        images: List,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Register a new user with multiple face samples.
        
        Args:
            user_id: Unique identifier for the user
            images: List of images (paths, PIL Images, or numpy arrays)
            metadata: Optional metadata about the user
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Check if user already exists
            if user_id in self.database:
                return False, f"User '{user_id}' already exists. Use update_user to modify."
            
            # Validate inputs
            if not user_id or not user_id.strip():
                return False, "User ID cannot be empty"
            
            if not images or len(images) == 0:
                return False, "At least one image is required for registration"
            
            # Extract features from all images
            print(f"Extracting features from {len(images)} images...")
            embeddings = []
            
            for i, image in enumerate(images):
                try:
                    embedding = self.feature_extractor.extract_features(image)
                    embeddings.append(embedding)
                    print(f"  Processed image {i+1}/{len(images)}")
                except Exception as e:
                    print(f"  Warning: Failed to process image {i+1}: {str(e)}")
                    continue
            
            if not embeddings:
                return False, "Failed to extract features from any image"
            
            # Calculate average embedding for robust representation
            avg_embedding = average_embeddings(embeddings)
            
            # Store user data
            user_data = {
                'user_id': user_id,
                'embedding': avg_embedding.tolist(),
                'num_samples': len(embeddings),
                'registration_date': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            self.database[user_id] = user_data
            self._save_database()
            
            return True, f"User '{user_id}' registered successfully with {len(embeddings)} samples"
        
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(
        self,
        image,
        return_confidence=True
    ) -> Tuple[Optional[str], float, str]:
        """
        Authenticate a user from a face image.
        
        Args:
            image: Image to authenticate (path, PIL Image, or numpy array)
            return_confidence: Whether to return confidence score
        
        Returns:
            Tuple of (user_id or None, confidence, message)
        """
        try:
            # Check if database is empty
            if not self.database:
                return None, 0.0, "No users registered in the database"
            
            # Extract features from input image
            query_embedding = self.feature_extractor.extract_features(image)
            
            # Compare with all registered users
            best_match_id = None
            best_similarity = -1.0
            
            for user_id, user_data in self.database.items():
                stored_embedding = np.array(user_data['embedding'])
                similarity = calculate_cosine_similarity(query_embedding, stored_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = user_id
            
            # Check if similarity meets threshold
            if best_similarity >= self.similarity_threshold:
                confidence_pct = best_similarity * 100
                message = f"Authenticated as '{best_match_id}' (confidence: {confidence_pct:.1f}%)"
                return best_match_id, best_similarity, message
            else:
                confidence_pct = best_similarity * 100
                message = f"Authentication failed. Best match: {confidence_pct:.1f}% (threshold: {self.similarity_threshold*100:.1f}%)"
                return None, best_similarity, message
        
        except Exception as e:
            return None, 0.0, f"Authentication error: {str(e)}"
    
    def verify_user(
        self,
        user_id: str,
        image
    ) -> Tuple[bool, float, str]:
        """
        Verify if an image matches a specific user (1:1 verification).
        
        Args:
            user_id: User ID to verify against
            image: Image to verify
        
        Returns:
            Tuple of (verified: bool, similarity: float, message: str)
        """
        try:
            # Check if user exists
            if user_id not in self.database:
                return False, 0.0, f"User '{user_id}' not found in database"
            
            # Extract features from input image
            query_embedding = self.feature_extractor.extract_features(image)
            
            # Get stored embedding
            stored_embedding = np.array(self.database[user_id]['embedding'])
            
            # Calculate similarity
            similarity = calculate_cosine_similarity(query_embedding, stored_embedding)
            
            # Verify against threshold
            if similarity >= self.similarity_threshold:
                confidence_pct = similarity * 100
                message = f"Verification successful (confidence: {confidence_pct:.1f}%)"
                return True, similarity, message
            else:
                confidence_pct = similarity * 100
                message = f"Verification failed (similarity: {confidence_pct:.1f}%, threshold: {self.similarity_threshold*100:.1f}%)"
                return False, similarity, message
        
        except Exception as e:
            return False, 0.0, f"Verification error: {str(e)}"
    
    def update_user(
        self,
        user_id: str,
        images: List,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Update an existing user's face template.
        
        Args:
            user_id: User ID to update
            images: New list of images
            metadata: Optional new metadata
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if user_id not in self.database:
                return False, f"User '{user_id}' not found"
            
            # Remove old user data
            del self.database[user_id]
            
            # Register with new data
            success, message = self.register_user(user_id, images, metadata)
            
            if success:
                return True, f"User '{user_id}' updated successfully"
            else:
                return False, f"Failed to update user: {message}"
        
        except Exception as e:
            return False, f"Update failed: {str(e)}"
    
    def delete_user(self, user_id: str) -> Tuple[bool, str]:
        """
        Delete a user from the database.
        
        Args:
            user_id: User ID to delete
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if user_id not in self.database:
                return False, f"User '{user_id}' not found"
            
            del self.database[user_id]
            self._save_database()
            
            return True, f"User '{user_id}' deleted successfully"
        
        except Exception as e:
            return False, f"Deletion failed: {str(e)}"
    
    def list_users(self) -> List[Dict]:
        """
        Get list of all registered users.
        
        Returns:
            List of user information dictionaries
        """
        users = []
        for user_id, user_data in self.database.items():
            users.append({
                'user_id': user_data['user_id'],
                'num_samples': user_data['num_samples'],
                'registration_date': user_data['registration_date'],
                'metadata': user_data.get('metadata', {})
            })
        return users
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        Get information about a specific user.
        
        Args:
            user_id: User ID
        
        Returns:
            User information dictionary or None
        """
        if user_id not in self.database:
            return None
        
        user_data = self.database[user_id]
        return {
            'user_id': user_data['user_id'],
            'num_samples': user_data['num_samples'],
            'registration_date': user_data['registration_date'],
            'metadata': user_data.get('metadata', {})
        }
    
    def set_threshold(self, threshold: float):
        """
        Update the similarity threshold.
        
        Args:
            threshold: New threshold value (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.similarity_threshold = threshold
        print(f"Threshold updated to {threshold}")
    
    def export_database(self, export_path: str, format='json'):
        """
        Export database to a file.
        
        Args:
            export_path: Path to export file
            format: Export format ('json' or 'pickle')
        """
        try:
            if format == 'json':
                # Convert numpy arrays to lists for JSON serialization
                export_data = {}
                for user_id, user_data in self.database.items():
                    export_data[user_id] = {
                        'user_id': user_data['user_id'],
                        'embedding': user_data['embedding'],
                        'num_samples': user_data['num_samples'],
                        'registration_date': user_data['registration_date'],
                        'metadata': user_data.get('metadata', {})
                    }
                
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format == 'pickle':
                with open(export_path, 'wb') as f:
                    pickle.dump(self.database, f)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            print(f"Database exported to {export_path}")
        
        except Exception as e:
            print(f"Export failed: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            'total_users': len(self.database),
            'similarity_threshold': self.similarity_threshold,
            'feature_dimension': self.feature_extractor.get_feature_dimension(),
            'use_gabor': self.use_gabor,
            'model_path': self.model_path,
            'database_path': self.database_path
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    print("Testing Face Authentication System...")
    
    # Initialize system
    auth_system = FaceAuthenticationSystem(
        model_path='nonexistent_model.h5',  # Will use dummy model
        similarity_threshold=0.6
    )
    
    # Print statistics
    stats = auth_system.get_statistics()
    print("\nSystem Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create test images
    test_img1 = np.random.rand(128, 128, 3) * 255
    test_img1 = test_img1.astype(np.uint8)
    
    test_img2 = np.random.rand(128, 128, 3) * 255
    test_img2 = test_img2.astype(np.uint8)
    
    # Register a user
    print("\n--- Testing Registration ---")
    success, message = auth_system.register_user(
        user_id='test_user_001',
        images=[test_img1, test_img1],  # Same image twice for demo
        metadata={'name': 'Test User', 'department': 'Engineering'}
    )
    print(message)
    
    # Authenticate
    print("\n--- Testing Authentication ---")
    user_id, confidence, message = auth_system.authenticate_user(test_img1)
    print(message)
    
    # Verify
    print("\n--- Testing Verification ---")
    verified, similarity, message = auth_system.verify_user('test_user_001', test_img1)
    print(message)
    
    # List users
    print("\n--- Registered Users ---")
    users = auth_system.list_users()
    for user in users:
        print(f"  {user['user_id']}: {user['num_samples']} samples")
