import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict
from typing import Optional

from datagenkit.utils.logging_utils import get_logger

logger = get_logger(__name__)

class EmbeddingExtractor:
    def __init__(self, max_cache_size=1000):
        """
        Initializes the EfficientNet_B0 model for feature extraction.
        Automatically selects GPU if available, otherwise runs on CPU.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device} for feature extraction")
        
        # Load EfficientNet_B0 pretrained on ImageNet
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)
        
        # Remove the final classification layer to get the feature representation
        self.model.classifier = torch.nn.Identity()
        
        self.model = self.model.to(self.device)
        self.model.eval()  # Put into inference mode
        
        # Standard normalization for ImageNet + Resizing to ensure all images fit
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # LRU Cache
        self.max_cache_size = max_cache_size
        self._cache = OrderedDict()

    def _hash_image(self, image: np.ndarray) -> str:
        # Simple fast hash based on sampling to avoid slow caching
        return str(hash(image.data.tobytes()[::100]))

    @torch.no_grad()
    def get_embedding(self, image: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """
        Extracts feature embedding for a single numpy RGB image (H, W, C).
        Returns a 1D numpy array representing the features.
        """
        if use_cache:
            img_hash = self._hash_image(image)
            if img_hash in self._cache:
                self._cache.move_to_end(img_hash) # LRU semantics
                return self._cache[img_hash]
                
        # Preprocess
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device) # Add batch dimension
        
        # Forward pass
        feature_tensor = self.model(input_batch)
        
        # Convert to numpy
        embedding = feature_tensor.cpu().numpy().flatten()
        
        if use_cache:
            self._cache[img_hash] = embedding
            if len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)
            
        return embedding
    
    def clear_cache(self):
        self._cache.clear()

_extractor_instance = None

def get_extractor() -> EmbeddingExtractor:
    """Returns a singleton of EmbeddingExtractor to save memory and initialization time across pipeline runs."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = EmbeddingExtractor()
    return _extractor_instance
