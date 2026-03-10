import os
import zipfile
from datagenkit.utils.logging_utils import get_logger

logger = get_logger(__name__)

def create_zip_archive(source_dir: str, output_filepath: str) -> bool:
    """
    Compresses a directory into a ZIP archive.
    Returns True if successful, False otherwise.
    """
    if not os.path.exists(source_dir):
        logger.error(f"Source directory {source_dir} does not exist.")
        return False
        
    try:
        with zipfile.ZipFile(output_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=source_dir)
                    zipf.write(file_path, arcname)
        logger.info(f"Successfully created ZIP archive at {output_filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to create ZIP archive: {e}")
        return False
