import os
from datagenkit.generator.dataset_builder import generate_dataset
from datagenkit.generator.generative_expansion import expand_dataset_with_ai
from datagenkit.generator.background_removal import isolate_subjects_in_directory
from datagenkit.utils.logging_utils import get_logger

logger = get_logger(__name__)

def run_datagen_pipeline(
    input_dir: str,
    output_dir: str,
    target_count: int = 50,
    similarity_threshold: float = 0.75,
    enable_isolation: bool = False,
    enable_ai: bool = False,
    hf_api_key: str = "",
    ai_prompt: str = "A highly detailed image",
    ai_num_new: int = 5,
    ai_strength: float = 0.6,
    enable_dynamic_prompts: bool = False,
    progress_callback=None
) -> dict:
    """
    Core entrypoint for the DatagenKit Python Library.
    Executes the Generative Expansion, Background Removal, and 
    MobileNetV2 Semantic Filtering Augmentation pipelines in sequence.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        # Phase 1: Hugging Face API Generation
        if enable_ai:
            logger.info("Connecting to Cloud AI Generator...")
            expand_dataset_with_ai(
                input_dir=input_dir,
                prompt=ai_prompt,
                api_key=hf_api_key,
                num_new_images=ai_num_new,
                strength=ai_strength,
                enhance_prompts=enable_dynamic_prompts,
                progress_callback=progress_callback
            )
            
        # Phase 2: U-2-Net Subject Isolation
        if enable_isolation:
            logger.info("Isolating salient subjects...")
            isolate_subjects_in_directory(
                input_dir=input_dir,
                progress_callback=progress_callback
            )
            
        # Phase 3: Dataset Augmentation Engine
        logger.info("Executing geometric/photometric augmentations and MobileNet filtering...")
        stats = generate_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            target_count=target_count,
            similarity_threshold=similarity_threshold,
            progress_callback=progress_callback
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"DatagenKit Pipeline crashed: {e}")
        raise
