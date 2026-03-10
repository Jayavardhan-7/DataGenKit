import albumentations as A
import numpy as np
import random
from typing import Optional

from datagenkit.generator.config import AUGMENTATION_PARAMS

def get_augmentation_pipeline(seed: Optional[int] = None) -> A.Compose:
    """
    Creates the Albumentations transformation pipeline.
    Uses optional seed for reproducible output if needed.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # Standard augmentation pipeline preserving semantic meaning
    transform = A.Compose([
        # Geometric - NO hardcoded resizing!
        A.HorizontalFlip(p=AUGMENTATION_PARAMS["p_flip"]),
        A.Rotate(limit=AUGMENTATION_PARAMS["rotate_limit"], p=AUGMENTATION_PARAMS["p_rotate"]),
        A.ShiftScaleRotate(
            shift_limit=AUGMENTATION_PARAMS["shift_limit"],
            scale_limit=AUGMENTATION_PARAMS["scale_limit"],
            rotate_limit=AUGMENTATION_PARAMS["rotate_limit"],
            p=AUGMENTATION_PARAMS["p_shift_scale_rotate"],
            border_mode=0
        ),
        # Advanced Geometric for semantic diversity
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        
        # Photometric
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=AUGMENTATION_PARAMS["p_brightness_contrast"]
        ),
        A.GaussianBlur(
            blur_limit=(3, 5), 
            p=AUGMENTATION_PARAMS["p_blur"]
        ),
        A.GaussNoise(
            p=AUGMENTATION_PARAMS["p_noise"]
        ),
    ])
    
    return transform

def augment_image(image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Apply standard DatagenKit smart augmentations to the given image.
    Expects RGB or RGBA numpy array.
    """
    pipeline = get_augmentation_pipeline(seed=seed)
    
    if image.shape[-1] == 4:
        # Separate alpha channel to avoid photometric distortions on transparency
        rgb = image[:, :, :3]
        alpha = image[:, :, 3]
        
        # Pass alpha as mask so geometric transforms apply to both
        augmented_res = pipeline(image=rgb, mask=alpha)
        
        # Stack back together
        aug_rgb = augmented_res["image"]
        aug_alpha = augmented_res["mask"]
        augmented = np.dstack((aug_rgb, aug_alpha))
    else:
        augmented = pipeline(image=image)["image"]
        
    return augmented
