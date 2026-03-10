import os
import uuid
from PIL import Image
from typing import Optional, Callable
from huggingface_hub import InferenceClient

from datagenkit.utils.logging_utils import get_logger

logger = get_logger(__name__)

def _generate_prompt_variations(base_prompt: str, n: int, api_key: str) -> list[str]:
    """Uses a free LLM on Hugging Face to generate N visual variations of a prompt."""
    try:
        client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct", token=api_key)
        
        system_msg = "You are a creative photography assistant. The user will give you a base image description. Generate rich, distinct visual scenarios for this description. Vary the lighting, environment, and camera angle. Output exactly one prompt per line. Do not include numbers, bullet points, introductory text, or explanations. Keep descriptions under 30 words."
        
        user_msg = f"Generate {n} distinct visual variations of this image prompt: '{base_prompt}'"
        
        res = client.chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=300,
            temperature=0.8
        )
        
        content = res.choices[0].message.content.strip()
        
        # Clean up output into a list of strings
        prompts = []
        for line in content.split('\n'):
            clean_line = line.strip()
            # Strip out leading numbers like "1. " or "- " if the model hallucinated them
            if clean_line:
                if len(clean_line) > 2 and clean_line[0].isdigit() and clean_line[1] in ('.', ')'):
                    clean_line = clean_line[2:].strip()
                elif clean_line.startswith('- ') or clean_line.startswith('* '):
                    clean_line = clean_line[2:].strip()
                prompts.append(clean_line)
                
        # Fallback if LLM failed to output exactly N lines
        while len(prompts) < n:
            prompts.append(base_prompt)
            
        return prompts[:n]
        
    except Exception as e:
        logger.error(f"Failed to generate dynamic prompts: {e}")
        # Fallback to base prompt repeated N times if LLM fails
        return [base_prompt] * n

def expand_dataset_with_ai(
    input_dir: str, 
    prompt: str, 
    api_key: str,
    num_new_images: int = 5, 
    strength: float = 0.6,
    enhance_prompts: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> int:
    """
    Reads existing images from `input_dir` and uses Hugging Face API (Img2Img)
    to generate `num_new_images` variants per seed based on the `prompt`. 
    New images are saved right back into `input_dir` to be picked up by the main generator.
    
    Args:
        input_dir: Directory containing original seeds.
        prompt: Text description of what the user wants to generate.
        api_key: Hugging Face API Token.
        num_new_images: Number of new structural variants per original seed.
        strength: How much the structural variant is allowed to deviate from the seed (0.0 to 1.0).
        progress_callback: UI progress callback.
        
    Returns:
        int: Number of new images successfully generated.
    """
    
    if not api_key:
        logger.error("No API key provided for generative expansion.")
        raise ValueError("Hugging Face API key is required to use AI Generative Expansion.")

    # 1. Gather all existing seed images
    seed_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            file_path = os.path.join(root, f)
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                seed_files.append(file_path)
                
    if not seed_files:
        logger.warning(f"No seed files found in {input_dir} for generative expansion.")
        return 0
        
    total_to_generate = len(seed_files) * num_new_images
    generated_count = 0
    
    # Hugging Face Inference API details
    # We use FLUX.1-schnell which is extremely fast, free, and high quality
    MODEL_ID = "black-forest-labs/FLUX.1-schnell"
    client = InferenceClient(MODEL_ID, token=api_key)
    
    # Pre-calculate prompt variations if requested
    if enhance_prompts:
        if progress_callback:
            progress_callback(0, total_to_generate, f"Brainstorming {total_to_generate} unique prompts using LLaMA...")
        dynamic_prompts = _generate_prompt_variations(prompt, total_to_generate, api_key)
    else:
        dynamic_prompts = [prompt] * total_to_generate
    
    # 2. Process each seed directory
    for file_path in seed_files:
        try:
            base_dir = os.path.dirname(file_path)
            # Default to jpg for FLUX
            ext = os.path.splitext(file_path)[1].lower()
            if not ext: ext = ".jpg"
            
            for i in range(num_new_images):
                if progress_callback:
                    msg = f"Requesting cloud AI generation {generated_count + 1}/{total_to_generate}..."
                    progress_callback(generated_count, total_to_generate, msg)
                    
                # Use Text-to-Image to generate a new structural variant based on the prompt
                result_image = client.text_to_image(prompt=prompt)
                
                # Save right next to original
                unique_name = f"ai_gen_{uuid.uuid4().hex[:8]}{ext}"
                out_path = os.path.join(base_dir, unique_name)
                
                if ext in [".png", ".webp"]:
                   result_image.save(out_path, format=ext[1:].upper())
                else:
                   result_image.save(out_path, format="JPEG", quality=95)
                   
                generated_count += 1
                
        except Exception as e:
            logger.error(f"Error expanding seed {file_path}: {e}")
            # If we hit API limits, stop trying the other images
            if "rate limit" in str(e).lower() or "402" in str(e):
                logger.error("Hugging Face API rate limit or credit limit reached.")
                break
            continue

    if progress_callback:
        progress_callback(total_to_generate, total_to_generate, f"AI generative expansion complete.")
            
    return generated_count
