from PIL import Image, ImageOps
import numpy as np
import os
from typing import Optional, Tuple

from datagenkit.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Supported formats based on requirements
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp'}
TARGET_SIZE = (224, 224)

def load_and_preprocess_image(file_path: str) -> Optional[np.ndarray]:
    """
    Safely loads an image, validates it, converts to RGB, and resizes to 224x224.
    Returns the processed image as a numpy array, or None if invalid.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        logger.warning(f"Unsupported format for {file_path}. Skipping.")
        return None

    try:
        # Load image safely using PIL
        with Image.open(file_path) as img:
            # Handle EXIF orientation tags before resizing/converting
            img = ImageOps.exif_transpose(img)
            
            # Convert to RGB (handles grayscale, RGBA)
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert('RGB')
                
            # Check for very small images
            if img.width < 10 or img.height < 10:
                logger.warning(f"Image {file_path} is too small ({img.size}). Skipping.")
                return None
                
            # Resize
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_arr = np.array(img)
            return img_arr
            
    except Exception as e:
        logger.error(f"Failed to load image {file_path}: {e}")
        return None

def normalize_for_model(img_arr: np.ndarray) -> np.ndarray:
    """
    Normalizes a 0-255 numpy image array to 0-1 range for model embeddings.
    Standard ImageNet normalization is applied inside the EmbeddingExtractor.
    This helper is just for basic bounds mapping if needed.
    """
    return img_arr.astype(np.float32) / 255.0
