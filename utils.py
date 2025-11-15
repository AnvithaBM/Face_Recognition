"""
Utility functions for face authentication system.
Includes image preprocessing, Gabor transform, and similarity calculations.
"""

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


# Gabor Transform configuration (matching notebook settings)
GABOR_KSIZE = 31
GABOR_SIGMA = 4.0
GABOR_THETA = np.pi / 4
GABOR_LAMBDA = 10.0
GABOR_GAMMA = 0.5


def apply_gabor_transform(image):
    """
    Apply Gabor transform to a grayscale image.
    
    Args:
        image: Grayscale image (normalized to [0, 1])
    
    Returns:
        Gabor filtered image
    """
    kernel = cv2.getGaborKernel(
        (GABOR_KSIZE, GABOR_KSIZE),
        GABOR_SIGMA,
        GABOR_THETA,
        GABOR_LAMBDA,
        GABOR_GAMMA,
        0,
        ktype=cv2.CV_32F
    )
    filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
    return np.abs(filtered)


def preprocess_image(image_input, target_size=(128, 128), use_gabor=True):
    """
    Preprocess an image for the face recognition model.
    
    Args:
        image_input: Can be a file path (str), PIL Image, or numpy array
        target_size: Tuple of (height, width) for resizing
        use_gabor: Whether to apply Gabor transform
    
    Returns:
        Preprocessed image array ready for model input
    """
    try:
        # Handle different input types
        if isinstance(image_input, str):
            # Load from file path
            img = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
            if img is None:
                img = np.array(Image.open(image_input))
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to numpy array
            img = np.array(image_input)
        elif isinstance(image_input, np.ndarray):
            # Already a numpy array
            img = image_input.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Handle different image formats
        if len(img.shape) == 2:  # Grayscale
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] > 3:  # Hyperspectral with multiple bands
            img = img[:, :, :3]
        
        # Convert BGR to RGB if needed (OpenCV loads as BGR)
        if isinstance(image_input, str) and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        if use_gabor:
            # Convert to grayscale for Gabor transform
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            gabor_feature = apply_gabor_transform(gray)
            # Concatenate Gabor feature as additional channel
            img = np.dstack([img, gabor_feature])
        
        return img
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def calculate_cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score (float between -1 and 1)
    """
    # Reshape if necessary
    emb1 = np.array(embedding1).reshape(1, -1)
    emb2 = np.array(embedding2).reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    
    return float(similarity)


def calculate_euclidean_distance(embedding1, embedding2):
    """
    Calculate Euclidean distance between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Euclidean distance (float)
    """
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    distance = np.linalg.norm(emb1 - emb2)
    
    return float(distance)


def average_embeddings(embeddings):
    """
    Calculate the average of multiple embeddings.
    
    Args:
        embeddings: List of embedding vectors
    
    Returns:
        Average embedding vector
    """
    if not embeddings:
        raise ValueError("Cannot average empty list of embeddings")
    
    embeddings_array = np.array(embeddings)
    average_emb = np.mean(embeddings_array, axis=0)
    
    return average_emb


def validate_image(image_path):
    """
    Validate if the given path points to a valid image file.
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False


def normalize_embedding(embedding):
    """
    Normalize an embedding vector to unit length.
    
    Args:
        embedding: Embedding vector
    
    Returns:
        Normalized embedding vector
    """
    emb = np.array(embedding)
    norm = np.linalg.norm(emb)
    
    if norm == 0:
        return emb
    
    return emb / norm
