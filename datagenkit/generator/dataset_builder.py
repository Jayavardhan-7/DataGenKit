import os
import random
import uuid
import time
from typing import Dict, Optional, Callable
from PIL import Image

from datagenkit.utils.logging_utils import get_logger
from datagenkit.utils.image_utils import load_and_preprocess_image
from datagenkit.generator.augmentations import augment_image
from datagenkit.generator.embeddings import EmbeddingExtractor
from datagenkit.generator.similarity import is_similar
from datagenkit.generator.config import DEFAULT_SIMILARITY_THRESHOLD, MAX_ATTEMPTS_MULTIPLIER

logger = get_logger(__name__)

def generate_dataset(
    input_dir: str,
    output_dir: str,
    target_count: int,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    seed: Optional[int] = None
) -> Dict:
    """
    Main orchestration engine to generate a synthetic image dataset.
    
    Args:
        input_dir: Directory containing seed images.
        output_dir: Output directory for generated images.
        target_count: Total number of output images desired (including seeds if preserved, though we just generate 'target' new images).
        similarity_threshold: Cosine similarity cutoff.
        progress_callback: Optional function(current_count, target_count, current_action) for UI.
        seed: Random seed for reproducibility.
        
    Returns:
        dict: Generation statistics.
    """
    start_time = time.time()
    
    if seed is not None:
        random.seed(seed)
        
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        "seeds_found": 0,
        "successfully_generated": 0,
        "discarded": 0,
        "total_attempts": 0,
        "time_taken_seconds": 0.0
    }
    
    # 1. Load seed images
    if progress_callback:
        progress_callback(0, target_count, "Loading seed images...")
        
    seed_images = []
    seed_filenames = []
    seed_rel_paths = []
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} not found.")
        return stats
        
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Make sure it looks like an image, though load_and_preprocess_image handles checking
            if os.path.isfile(file_path):
                img = load_and_preprocess_image(file_path)
                if img is not None:
                    seed_images.append(img)
                    seed_filenames.append(file)
                    # Preserve relative path for subdirectories
                    rel_path = os.path.relpath(root, input_dir)
                    seed_rel_paths.append(rel_path)
                
    stats["seeds_found"] = len(seed_images)
    if not seed_images:
        logger.error("No valid seed images found in input directory.")
        return stats
        
    logger.info(f"Loaded {len(seed_images)} seed images successfully.")
    
    # 2. Extract embeddings for seeds
    if progress_callback:
        progress_callback(0, target_count, "Extracting features from seeds...")
        
    from datagenkit.generator.embeddings import get_extractor
    extractor = get_extractor()
    seed_embeddings = []
    
    for img in seed_images:
        emb = extractor.get_embedding(img)
        seed_embeddings.append(emb)
        
    # 3. Generation Loop
    import concurrent.futures
    import threading
    from datagenkit.generator.config import MAX_WORKERS

    current_generated_count = 0
    max_total_attempts = target_count * MAX_ATTEMPTS_MULTIPLIER
    
    if progress_callback:
        progress_callback(0, target_count, f"Generating {target_count} images...")
        
    lock = threading.Lock()
    
    def process_attempt(attempt_seed: Optional[int] = None):
        # Pick a random seed
        seed_idx = random.randint(0, len(seed_images) - 1)
        base_img = seed_images[seed_idx]
        base_filename = seed_filenames[seed_idx]
        base_rel_path = seed_rel_paths[seed_idx]
        
        ext = os.path.splitext(base_filename)[1].lower()
        if not ext:
            ext = ".jpg"
            
        # Augment
        aug_img = augment_image(base_img, seed=attempt_seed) 
        
        # Extract features (use cache false for augmentations)
        aug_emb = extractor.get_embedding(aug_img, use_cache=False) 
        
        # Compare specifically against the source seed
        is_sim = is_similar(aug_emb, [seed_embeddings[seed_idx]], threshold=similarity_threshold)
        return is_sim, aug_img, ext, base_rel_path

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = set()
        
        # Initial submission
        for _ in range(min(MAX_WORKERS, max_total_attempts)):
            func_seed = seed + stats["total_attempts"] if seed is not None else None
            stats["total_attempts"] += 1
            futures.add(executor.submit(process_attempt, func_seed))

        while futures and current_generated_count < target_count:
            # Wait for at least one to complete
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            
            for future in done:
                try:
                    is_sim, aug_img, ext, rel_path = future.result()
                    
                    if is_sim and current_generated_count < target_count:
                        # Save, preserving subdirectories
                        save_dir = os.path.join(output_dir, rel_path)
                        if rel_path != "." and not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                            
                        unique_filename = f"gen_{uuid.uuid4().hex[:8]}{ext}"
                        out_path = os.path.join(save_dir, unique_filename)
                        
                        # Convert back to PIL for saving, keeping RGBA if applicable
                        mode = "RGBA" if aug_img.shape[-1] == 4 else "RGB"
                        pil_img = Image.fromarray(aug_img, mode=mode)
                        
                        if ext in [".png", ".webp"]:
                            pil_img.save(out_path, format=ext[1:].upper())
                        else:
                            pil_img.save(out_path, format="JPEG", quality=95)
                            
                        with lock:
                            current_generated_count += 1
                            stats["successfully_generated"] += 1
                            if progress_callback:
                                progress_callback(current_generated_count, target_count, f"Generated {current_generated_count}/{target_count} images")
                    else:
                        with lock:
                            stats["discarded"] += 1
                except Exception as e:
                    logger.error(f"Error in generation thread: {e}")
                    with lock:
                        stats["total_attempts"] -= 1 # Re-attempt on generic failure
                
                # Submit a new job if we still need more images and haven't hit the limit
                with lock:
                    if current_generated_count < target_count and stats["total_attempts"] < max_total_attempts:
                        func_seed = seed + stats["total_attempts"] if seed is not None else None
                        stats["total_attempts"] += 1
                        futures.add(executor.submit(process_attempt, func_seed))
                    
    if stats["total_attempts"] >= max_total_attempts and current_generated_count < target_count:
        logger.warning(f"Reached maximum attempts ({max_total_attempts}). Stopping generation early.")
            
    stats["time_taken_seconds"] = round(time.time() - start_time, 2)
    logger.info(f"Generation complete: {stats}")
    
    return stats
