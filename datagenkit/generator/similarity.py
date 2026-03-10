import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

from datagenkit.generator.config import DEFAULT_SIMILARITY_THRESHOLD

def is_similar(aug_emb: np.ndarray, original_embs: List[np.ndarray], threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> bool:
    """
    Computes cosine similarity of an augmented image's embedding against
    seed embeddings.
    
    Returns True if the maximum similarity to ANY provided seed is >= threshold.
    """
    if len(original_embs) == 0:
        return False
        
    # Reshape for sklearn
    aug_emb_reshaped = aug_emb.reshape(1, -1)
    
    # We can stack original embeddings into a single matrix
    # shape (N_seeds, Embedding_dim)
    orig_embs_matrix = np.vstack(original_embs)
    
    # Compute similarities (returns matrix of shape (1, N_seeds))
    similarities = cosine_similarity(aug_emb_reshaped, orig_embs_matrix)
    
    # Check max similarity against threshold
    max_sim = np.max(similarities)
    
    return max_sim >= threshold
