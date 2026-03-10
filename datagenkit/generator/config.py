# Configurable hyperparameters for generation

# Target size for processing (MobileNetV2 expects 224x224 usually, though it can adapt)
IMAGE_SIZE = (224, 224)

# Similarity settings
DEFAULT_SIMILARITY_THRESHOLD = 0.75

# Augmentation strengths
AUGMENTATION_PARAMS = {
    "rotate_limit": 30,
    "p_flip": 0.5,
    "p_rotate": 0.8,
    "p_crop": 0.5,
    "p_shift_scale_rotate": 0.5,
    "p_brightness_contrast": 0.5,
    "p_blur": 0.3,
    "p_noise": 0.3,
    
    # Mild shift scale rotate
    "shift_limit": 0.0625,
    "scale_limit": 0.1,
}

# Maximum attempts to find a valid similar image before giving up for a seed
MAX_ATTEMPTS_MULTIPLIER = 10

# Maximum number of threads for dataset generation parallelization
MAX_WORKERS = 4
