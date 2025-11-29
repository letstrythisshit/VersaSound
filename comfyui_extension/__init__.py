"""
VersaSound ComfyUI Extension
Universal video-to-audio generation for ComfyUI

A complete, production-ready extension for generating synchronized audio
from video content using state-of-the-art AI models.
"""

__version__ = "1.0.0"
__author__ = "VersaSound Team"
__license__ = "MIT"

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import nodes
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    logger.info(f"VersaSound v{__version__} loaded successfully")
    logger.info(f"Registered {len(NODE_CLASS_MAPPINGS)} nodes")

    # Check if models are downloaded
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    if not checkpoint_dir.exists() or not list(checkpoint_dir.glob("*.safetensors")):
        logger.warning("=" * 60)
        logger.warning("VersaSound: Model checkpoints not found")
        logger.warning("=" * 60)
        logger.warning("Please run the installation script:")
        logger.warning("  cd comfyui_extension")
        logger.warning("  python install.py")
        logger.warning("")
        logger.warning("This will download required models (~2GB)")
        logger.warning("=" * 60)

    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

except Exception as e:
    logger.error(f"Error loading VersaSound: {e}")
    logger.error("Please check installation and dependencies")
    raise
