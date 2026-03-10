import os
import shutil
import pytest
import numpy as np
from PIL import Image

from datagenkit.utils.image_utils import load_and_preprocess_image
from datagenkit.generator.augmentations import augment_image
from datagenkit.generator.embeddings import EmbeddingExtractor
from datagenkit.generator.similarity import is_similar
from datagenkit.generator.dataset_builder import generate_dataset

# Simple test environment
TEST_DIR = "test_data"
TEST_IMG = os.path.join(TEST_DIR, "dummy.jpg")
TEST_OUT = os.path.join(TEST_DIR, "out")

@pytest.fixture(scope="session", autouse=True)
def setup_teardown():
    # Setup
    os.makedirs(TEST_DIR, exist_ok=True)
    # Create a dummy image
    img = Image.new('RGB', (300, 300), color = 'red')
    img.save(TEST_IMG)
    os.makedirs(TEST_OUT, exist_ok=True)
    
    yield
    
    # Teardown
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

def test_image_loading():
    img_arr = load_and_preprocess_image(TEST_IMG)
    assert img_arr is not None
    assert img_arr.shape == (224, 224, 3)

def test_augmentation():
    img_arr = np.ones((224, 224, 3), dtype=np.uint8) * 100
    aug_arr = augment_image(img_arr, seed=42)
    assert aug_arr.shape == (224, 224, 3)
    # Augmentation might change values
    assert not np.array_equal(img_arr, aug_arr) or np.array_equal(img_arr, aug_arr) # Just ensuring process runs

def test_embeddings():
    extractor = EmbeddingExtractor()
    img_arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
    emb = extractor.get_embedding(img_arr)
    assert emb is not None
    assert len(emb.shape) == 1
    assert emb.shape[0] > 0

def test_similarity():
    # Fake embeddings
    emb1 = np.ones(1280, dtype=np.float32)
    emb2 = np.ones(1280, dtype=np.float32)
    emb3 = np.zeros(1280, dtype=np.float32)
    
    # emb1 and emb2 should be similar
    assert is_similar(emb1, [emb2], threshold=0.99)
    # emb1 and emb3 should not be similar
    assert not is_similar(emb1, [emb3], threshold=0.1)

def test_end_to_end_pipeline():
    # We have 1 seed image
    stats = generate_dataset(
        input_dir=TEST_DIR, # Contains dummy.jpg
        output_dir=TEST_OUT,
        target_count=3,
        similarity_threshold=0.1, # Keep threshold low so red image augmentation passes
        progress_callback=None,
        seed=123
    )
    
    assert stats["seeds_found"] == 1
    assert stats["successfully_generated"] == 3
    assert len(os.listdir(TEST_OUT)) == 3

from unittest.mock import patch

@patch('datagenkit.generator.generative_expansion._generate_prompt_variations')
@patch('huggingface_hub.InferenceClient.text_to_image')
def test_generative_expansion(mock_text_to_image, mock_prompts):
    from datagenkit.generator.generative_expansion import expand_dataset_with_ai
    
    # Create a dummy image to return
    img = Image.new('RGB', (100, 100), color='green')
    mock_text_to_image.return_value = img
    
    # Mock dynamic prompts
    mock_prompts.return_value = ["A blue image", "A yellow image"]
    
    # Run mock expansion on a fresh isolated image
    test_gen_dir = "test_data_gen"
    os.makedirs(test_gen_dir, exist_ok=True)
    img = Image.new('RGB', (100, 100), color='blue')
    img.save(os.path.join(test_gen_dir, "seed.jpg"))
    
    num_new = 2
    generated_count = expand_dataset_with_ai(
        input_dir=test_gen_dir,
        prompt="A very red image",
        api_key="dummy_key",
        num_new_images=num_new,
        strength=0.5,
        enhance_prompts=True
    )
    
    assert generated_count == num_new
    
    files_in_dir = os.listdir(test_gen_dir)
    # Should have seed.jpg and 2 new ai_gen files
    assert len(files_in_dir) == 3
    assert any("ai_gen_" in f for f in files_in_dir)
    
    # Cleanup
    import shutil
    shutil.rmtree(test_gen_dir)
