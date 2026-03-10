import os
from PIL import Image
from typing import Optional, Callable
import rembg

from datagenkit.utils.logging_utils import get_logger

logger = get_logger(__name__)

def isolate_subjects_in_directory(
    input_dir: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> int:
    """
    Iterates through all images in the `input_dir` and uses `rembg` to remove
    their backgrounds, returning an image with a purely transparent alpha channel.
    The original file is overwritten as a natively transparent `.png`.
    
    Args:
        input_dir: Directory containing the images to process.
        progress_callback: UI progress callback.
        
    Returns:
        int: Number of images successfully isolated.
    """
    
    processed_count = 0
    target_files = []
    
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                target_files.append(os.path.join(root, f))
                
    if not target_files:
        logger.warning(f"No files found in {input_dir} for background removal.")
        return 0
        
    total_files = len(target_files)
    
    # Initialize a new rembg session explicitly to manage memory if needed,
    # though remove() handles it automatically by default.
    from rembg import new_session
    session = new_session()
    
    for idx, file_path in enumerate(target_files):
        if progress_callback:
            progress_callback(idx, total_files, f"Isolating Subject {idx + 1}/{total_files}...")
            
        try:
            # We must convert to RGB/RGBA so Pillow/rembg don't choke on palettes or grayscale
            input_image = Image.open(file_path).convert('RGB')
            subject = rembg.remove(input_image, session=session)
            
            # Save the new transparent subject out
            base = os.path.splitext(file_path)[0]
            new_path = f"{base}.png"
            subject.save(new_path, format="PNG")
            
            # Delete original if it wasn't a PNG initially to prevent duplication
            if file_path != new_path:
                os.remove(file_path)
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to remove background from {file_path}: {e}")
            continue
            
    if progress_callback:
        progress_callback(total_files, total_files, "Subject Isolation Complete.")
        
    return processed_count
