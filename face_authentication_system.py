"""
Hyperspectral Face Authentication System

This module provides a complete authentication system for hyperspectral face recognition,
including user enrollment and verification capabilities.
"""

import numpy as np
import pickle
import os
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class HyperspectralFaceAuthenticator:
    """
    A face authentication system using hyperspectral face recognition.
    
    This class provides methods for:
    - Loading pre-trained embedding models
    - Enrolling new users into the database
    - Authenticating users against enrolled embeddings
    - Managing the user database
    """
    
    def __init__(self, 
                 model_path: str = 'hyperspectral_embedding_model.h5',
                 database_path: str = 'user_database.pkl',
                 similarity_metric: str = 'cosine',
                 threshold: float = 0.5):
        """
        Initialize the authentication system.
        
        Args:
            model_path: Path to the trained embedding model (HDF5 file)
            database_path: Path to the user database file (pickle)
            similarity_metric: Metric for comparison ('cosine' or 'euclidean')
            threshold: Acceptance threshold for authentication
                      For cosine: higher is more similar (typical: 0.5-0.8)
                      For euclidean: lower is more similar (typical: 0.5-2.0)
        """
        self.model_path = model_path
        self.database_path = database_path
        self.similarity_metric = similarity_metric
        self.threshold = threshold
        
        # Load the embedding model
        self.model = self._load_model()
        
        # Load or initialize user database
        self.user_database = self._load_database()
        
    def _load_model(self) -> tf.keras.Model:
        """Load the pre-trained embedding model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Please train the model first using the training notebook."
            )
        
        print(f"Loading model from {self.model_path}...")
        # Use safe_mode=False to allow loading Lambda layers with custom functions
        # This is needed for L2 normalization layer
        model = tf.keras.models.load_model(self.model_path, safe_mode=False)
        print(f"Model loaded successfully. Input shape: {model.input_shape}")
        return model
    
    def _load_database(self) -> Dict:
        """Load the user database or create a new one if it doesn't exist."""
        if os.path.exists(self.database_path):
            print(f"Loading user database from {self.database_path}...")
            with open(self.database_path, 'rb') as f:
                database = pickle.load(f)
            print(f"Loaded {len(database)} users from database.")
            return database
        else:
            print("No existing database found. Creating new database.")
            return {}
    
    def _save_database(self):
        """Save the user database to disk."""
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.user_database, f)
        print(f"Database saved to {self.database_path}")
    
    def extract_embedding(self, hyperspectral_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a hyperspectral face image.
        
        Args:
            hyperspectral_image: Hyperspectral image array of shape (H, W, C)
        
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        # Ensure image has batch dimension
        if len(hyperspectral_image.shape) == 3:
            hyperspectral_image = np.expand_dims(hyperspectral_image, axis=0)
        
        # Extract embedding
        embedding = self.model.predict(hyperspectral_image, verbose=0)
        
        # Return as 1D array
        return embedding.squeeze()
    
    def enroll_user(self, 
                    user_id: str, 
                    hyperspectral_images: List[np.ndarray],
                    overwrite: bool = False) -> bool:
        """
        Enroll a new user into the authentication system.
        
        Args:
            user_id: Unique identifier for the user
            hyperspectral_images: List of hyperspectral face images for enrollment
                                 (multiple images improve robustness)
            overwrite: Whether to overwrite existing user data
        
        Returns:
            True if enrollment successful, False otherwise
        """
        # Check if user already exists
        if user_id in self.user_database and not overwrite:
            print(f"User '{user_id}' already exists. Use overwrite=True to update.")
            return False
        
        # Extract embeddings from all provided images
        embeddings = []
        for img in hyperspectral_images:
            embedding = self.extract_embedding(img)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # Store user data
        self.user_database[user_id] = {
            'embeddings': embeddings,
            'num_samples': len(embeddings),
            'mean_embedding': np.mean(embeddings, axis=0)
        }
        
        # Save database
        self._save_database()
        
        print(f"✓ User '{user_id}' enrolled successfully with {len(embeddings)} samples.")
        return True
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score
        """
        # Ensure 2D shape for sklearn functions
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)
        
        if self.similarity_metric == 'cosine':
            # Cosine similarity (higher is more similar)
            similarity = cosine_similarity(emb1, emb2)[0, 0]
        elif self.similarity_metric == 'euclidean':
            # Euclidean distance (lower is more similar)
            # Convert to similarity by negating
            distance = euclidean_distances(emb1, emb2)[0, 0]
            similarity = -distance
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarity
    
    def authenticate(self, 
                    hyperspectral_image: np.ndarray,
                    user_id: Optional[str] = None) -> Tuple[bool, str, float]:
        """
        Authenticate a user based on their hyperspectral face image.
        
        Args:
            hyperspectral_image: Hyperspectral face image to authenticate
            user_id: Optional specific user to authenticate against (1:1 verification)
                    If None, performs 1:N identification
        
        Returns:
            Tuple of (authenticated, matched_user_id, confidence_score)
        """
        if len(self.user_database) == 0:
            print("No users enrolled in database.")
            return False, None, 0.0
        
        # Extract embedding from input image
        query_embedding = self.extract_embedding(hyperspectral_image)
        
        # Case 1: 1:1 Verification (specific user)
        if user_id is not None:
            if user_id not in self.user_database:
                print(f"User '{user_id}' not found in database.")
                return False, None, 0.0
            
            user_data = self.user_database[user_id]
            mean_embedding = user_data['mean_embedding']
            
            similarity = self._compute_similarity(query_embedding, mean_embedding)
            
            # Apply threshold
            if self.similarity_metric == 'cosine':
                authenticated = similarity >= self.threshold
            else:  # euclidean (similarity is negative distance)
                authenticated = similarity >= -self.threshold
            
            return authenticated, user_id if authenticated else None, similarity
        
        # Case 2: 1:N Identification (find best match)
        best_match = None
        best_similarity = -np.inf
        
        for uid, user_data in self.user_database.items():
            mean_embedding = user_data['mean_embedding']
            similarity = self._compute_similarity(query_embedding, mean_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = uid
        
        # Apply threshold
        if self.similarity_metric == 'cosine':
            authenticated = best_similarity >= self.threshold
        else:  # euclidean
            authenticated = best_similarity >= -self.threshold
        
        return authenticated, best_match if authenticated else None, best_similarity
    
    def remove_user(self, user_id: str) -> bool:
        """
        Remove a user from the database.
        
        Args:
            user_id: User identifier to remove
        
        Returns:
            True if user was removed, False if user not found
        """
        if user_id in self.user_database:
            del self.user_database[user_id]
            self._save_database()
            print(f"✓ User '{user_id}' removed from database.")
            return True
        else:
            print(f"User '{user_id}' not found in database.")
            return False
    
    def list_users(self) -> List[str]:
        """
        List all enrolled users.
        
        Returns:
            List of user IDs
        """
        return list(self.user_database.keys())
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        Get information about a specific user.
        
        Args:
            user_id: User identifier
        
        Returns:
            Dictionary with user information or None if not found
        """
        if user_id in self.user_database:
            user_data = self.user_database[user_id]
            return {
                'user_id': user_id,
                'num_samples': user_data['num_samples'],
                'embedding_dim': len(user_data['mean_embedding'])
            }
        return None
    
    def update_threshold(self, new_threshold: float):
        """
        Update the authentication threshold.
        
        Args:
            new_threshold: New threshold value
        """
        self.threshold = new_threshold
        print(f"Threshold updated to {new_threshold}")
    
    def export_database(self, export_path: str):
        """
        Export the user database to a different location.
        
        Args:
            export_path: Path to export the database
        """
        with open(export_path, 'wb') as f:
            pickle.dump(self.user_database, f)
        print(f"Database exported to {export_path}")
    
    def import_database(self, import_path: str, merge: bool = False):
        """
        Import a user database from a file.
        
        Args:
            import_path: Path to the database file to import
            merge: If True, merge with existing database; if False, replace
        """
        with open(import_path, 'rb') as f:
            imported_db = pickle.load(f)
        
        if merge:
            self.user_database.update(imported_db)
            print(f"Merged {len(imported_db)} users into database.")
        else:
            self.user_database = imported_db
            print(f"Replaced database with {len(imported_db)} users.")
        
        self._save_database()


def load_hyperspectral_image(image_path: str, 
                             target_size: Tuple[int, int] = (128, 128),
                             normalize: bool = True) -> np.ndarray:
    """
    Load a hyperspectral image from file.
    
    Args:
        image_path: Path to the hyperspectral image file (.npy or .mat)
        target_size: Target size (height, width) to resize to
        normalize: Whether to normalize the image
    
    Returns:
        Hyperspectral image array of shape (H, W, C)
    """
    if image_path.endswith('.npy'):
        img = np.load(image_path)
    elif image_path.endswith('.mat'):
        from scipy.io import loadmat
        data = loadmat(image_path)
        # Adjust key based on your .mat file structure
        img = data.get('hyperspectral_image', data[list(data.keys())[-1]])
    else:
        raise ValueError(f"Unsupported file format: {image_path}")
    
    # Resize if needed
    if img.shape[:2] != target_size:
        from scipy.ndimage import zoom
        zoom_factors = (target_size[0]/img.shape[0], 
                       target_size[1]/img.shape[1], 1)
        img = zoom(img, zoom_factors, order=1)
    
    # Normalize
    if normalize:
        img = (img - img.mean()) / (img.std() + 1e-7)
    
    return img


def create_synthetic_hyperspectral_image(size: Tuple[int, int] = (128, 128),
                                         channels: int = 33) -> np.ndarray:
    """
    Create a synthetic hyperspectral image for testing.
    
    Args:
        size: Image size (height, width)
        channels: Number of spectral channels
    
    Returns:
        Synthetic hyperspectral image
    """
    img = np.random.randn(*size, channels).astype(np.float32)
    # Normalize
    img = (img - img.mean()) / (img.std() + 1e-7)
    return img


if __name__ == "__main__":
    print("Hyperspectral Face Authentication System")
    print("=" * 50)
    print("\nThis module provides authentication capabilities.")
    print("Please use the example scripts or notebooks to interact with the system.")
