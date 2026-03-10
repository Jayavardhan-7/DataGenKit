from datagenkit.pipeline import run_datagen_pipeline
import os

# Create a dummy image
from PIL import Image
os.makedirs("demo_seeds", exist_ok=True)
img = Image.new('RGB', (100, 100), color='blue')
img.save("demo_seeds/blue_square.jpg")

print("Running DatagenKit Pipeline programmatically...")
stats = run_datagen_pipeline(
    input_dir="demo_seeds",
    output_dir="demo_output",
    target_count=5, # Just generate 5 images for a quick test
    similarity_threshold=0.1
)

print("\n--- SYNTHESIS COMPLETE ---")
print("Statistics:", stats)
print("Files generated in 'demo_output':", os.listdir("demo_output"))
