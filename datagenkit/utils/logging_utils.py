import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger with standard formatting."""
    logger = logging.getLogger(name)
    
    # Only configure if no handlers are present to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Create console handler with formatting
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
