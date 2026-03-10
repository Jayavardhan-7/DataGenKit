# DatagenKit

DatagenKit is a robust, production-quality Image Dataset Generator Python Library and Command Line Tool. It effortlessly expands a small set of user-provided seed images into a larger, high-quality synthetic dataset using LLM-guided Generative Expansion, advanced Background Removal, lightweight geometric augmentations, and MobileNetV2 feature-based semantic filtering.

## 🚀 Features
- **Generative AI Expansion**: Seamlessly connect to Hugging Face Cloud APIs (FLUX) to predictably synthesize entirely new structural variants from your base seeds.
- **Dynamic Prompt Engineering**: Intercept your base textual prompts and use LLaMA-3.2 to rewrite them into dozens of unique, rich scenarios automatically.
- **Intelligent Subject Isolation**: Uses U-2-Net (`rembg`) to optionally strip out backgrounds, ensuring your downstream models learn the subject and not the background (preventing shortcut learning).
- **Smart Augmentations**: Uses `albumentations` for geometric and photometric transformations natively on 4-channel transparent PNGs.
- **Pretrained Filtering**: Uses MobileNetV2 for feature extraction and Cosine Similarity to discard any unrealistic or overly-distorted augmentations.
- **Headless CLI & Library**: Designed natively for automated MLOps pipelines. Completely terminal driven with a flexible Python API.

---

## 📦 Installation

To use DatagenKit globally as a CLI tool or import it into your Python environment, install it via pip:

```sh
# Clone the repository and install it globally
git clone <your-repo-url>
cd datagenkit
pip install -e .
```

*Note: DatagenKit installs `datagenkit` into your PATH automatically.*

---

## 💻 Usage: Command Line Interface (CLI)

DatagenKit provides a rich, deeply configurable CLI. You can view all available arguments anytime:
```sh
datagenkit --help
```

### Basic Generation (Augmentations & Filtering Only)
```sh
datagenkit --input-dir my_seed_images/ --output-dir generated_dataset/ --target-count 250
```

### Advanced Pipeline (AI Expansion + Background Removal)
```sh
datagenkit -i my_seed_images/ -o final_dataset/ -n 250 \
  --enable-isolation \
  --enable-ai \
  --hf-api-key "hf_your_token_here" \
  --ai-prompt "A highly detailed cat sitting on a rug" \
  --enable-dynamic-prompts
```

---

## 🐍 Usage: Python API

If you are building your own Python scripts, custom data-loaders, or Jupyter Notebooks, you can import DatagenKit directly:

```python
from datagenkit.pipeline import run_datagen_pipeline

# Run the complete, end-to-end dataset synthesizer
stats = run_datagen_pipeline(
    input_dir="my_seed_images/",
    output_dir="final_dataset/",
    target_count=250,
    similarity_threshold=0.75,
    
    # Advanced ML Features
    enable_isolation=True,      # Automatically remove backgrounds natively
    enable_ai=True,             # Generate new variants via FLUX API
    hf_api_key="hf_xxx",        # Your HuggingFace Token
    ai_prompt="A photo of a dog",
    enable_dynamic_prompts=True # Let LLaMA-3.2 enrich the prompt into variations
)

print(f"Generation Complete! Stats: {stats}")
```

## 🛠 Prerequisites

- Python >= 3.8
- A Hugging Face account & Access Token (Only required if you enable AI Expansion capabilities).
