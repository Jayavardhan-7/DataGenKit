import argparse
import sys
import os
from datagenkit.pipeline import run_datagen_pipeline
from datagenkit.utils.logging_utils import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="DatagenKit: Synthesize, augment, and filter semantic image datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir", "-i", type=str, required=True,
        help="Path to the directory containing seed images or class subfolders."
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, required=True,
        help="Path where the synthesized datast should be saved."
    )
    parser.add_argument(
        "--target-count", "-n", type=int, default=50,
        help="Total number of images to synthesize."
    )
    parser.add_argument(
        "--similarity-threshold", "-s", type=float, default=0.75,
        help="Cosine similarity threshold (0.0 to 1.0) for the MobileNetV2 semantic filter."
    )

    # Advanced Machine Learning Features
    ml_group = parser.add_argument_group("Advanced ML Configurations")
    ml_group.add_argument(
        "--enable-isolation", action="store_true",
        help="Automatically perfectly strip the background from all input images using U-2-Net (rembg) prior to augmentation to prevent shortcut learning."
    )
    ml_group.add_argument(
        "--enable-ai", action="store_true",
        help="Use a free Hugging Face API (FLUX.1-schnell model) to predictably expand the pure structural variety of your initial dataset seeds."
    )
    ml_group.add_argument(
        "--hf-api-key", type=str, default="",
        help="Your Hugging Face Access Token. Required if --enable-ai is utilized."
    )
    ml_group.add_argument(
        "--ai-prompt", type=str, default="A highly detailed image",
        help="The base text prompt describing your target seeds for the AI generative expansion."
    )
    ml_group.add_argument(
        "--ai-num-new", type=int, default=5,
        help="The number of AI generated structural variants to create per seed image."
    )
    ml_group.add_argument(
        "--ai-strength", type=float, default=0.6,
        help="Obsolete text-to-image variance strength (reserved for legacy pipelines)."
    )
    ml_group.add_argument(
        "--enable-dynamic-prompts", action="store_true",
        help="Use LLaMA-3.2 to dynamically rewrite and enrich your base --ai-prompt into dozens of totally unique scenarios for the FLUX engine."
    )

    args = parser.parse_args()

    # Validations
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    if args.enable_ai and not args.hf_api_key:
        logger.error("You must provide an --hf-api-key if --enable-ai is utilized.")
        sys.exit(1)

    logger.info("Starting DatagenKit Pipeline...")
    try:
        stats = run_datagen_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_count=args.target_count,
            similarity_threshold=args.similarity_threshold,
            enable_isolation=args.enable_isolation,
            enable_ai=args.enable_ai,
            hf_api_key=args.hf_api_key,
            ai_prompt=args.ai_prompt,
            ai_num_new=args.ai_num_new,
            ai_strength=args.ai_strength,
            enable_dynamic_prompts=args.enable_dynamic_prompts
        )
        logger.info(f"Pipeline Completed Successfully: {stats}")
    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
