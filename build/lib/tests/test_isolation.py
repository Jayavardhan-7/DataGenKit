import pytest
import os
import shutil
from PIL import Image

def test_subject_isolation():
    from datagenkit.generator.background_removal import isolate_subjects_in_directory
    
    test_iso_dir = "test_iso_gen"
    if os.path.exists(test_iso_dir):
        shutil.rmtree(test_iso_dir)
    os.makedirs(test_iso_dir, exist_ok=True)
    
    # Create a dummy image (green square inside a red background) to test removal
    # Rembg might just blank a pure color block, but we can test the file handling logic
    img = Image.new('RGB', (100, 100), color='red')
    img_path = os.path.join(test_iso_dir, "dummy.jpg")
    img.save(img_path)
    
    count = isolate_subjects_in_directory(test_iso_dir)
    
    # Check that it processed 1 file
    assert count == 1
    
    # Check that the file was converted to PNG
    assert not os.path.exists(img_path)
    png_path = os.path.join(test_iso_dir, "dummy.png")
    assert os.path.exists(png_path)
    
    # Check that the new image has an alpha channel (RGBA)
    with Image.open(png_path) as result_img:
        assert result_img.mode == 'RGBA'
    
    # Cleanup
    shutil.rmtree(test_iso_dir)
