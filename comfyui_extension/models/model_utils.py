"""
Model utility functions for VersaSound
Helper functions for model loading, inference, and management
"""

import torch
import logging
from typing import Dict, Optional, Any
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def load_model_config(config_path: Optional[str] = None) -> Dict:
    """
    Load model configuration from YAML file

    Args:
        config_path: Path to config file (None = use default)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default config
        config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def get_checkpoint_path(
    model_name: str,
    checkpoint_dir: Optional[str] = None
) -> Path:
    """
    Get path to model checkpoint

    Args:
        model_name: Name of model ('visual_encoder', 'audio_generator', etc.)
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to checkpoint file
    """
    if checkpoint_dir is None:
        checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_dir)

    checkpoint_file = checkpoint_dir / f"{model_name}.safetensors"

    if not checkpoint_file.exists():
        # Try .pt extension
        checkpoint_file = checkpoint_dir / f"{model_name}.pt"

    if not checkpoint_file.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_file}")
        return None

    return checkpoint_file


def download_checkpoint(
    model_name: str,
    checkpoint_dir: Optional[str] = None,
    base_url: str = "https://huggingface.co/versasound"
) -> bool:
    """
    Download model checkpoint from Hugging Face

    Args:
        model_name: Name of model
        checkpoint_dir: Directory to save checkpoint
        base_url: Base URL for downloads

    Returns:
        True if successful, False otherwise
    """
    try:
        import requests
        from tqdm import tqdm

        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
        else:
            checkpoint_dir = Path(checkpoint_dir)

        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Construct URL
        url = f"{base_url}/{model_name}/resolve/main/model.safetensors"
        destination = checkpoint_dir / f"{model_name}.safetensors"

        logger.info(f"Downloading {model_name} from {url}")

        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"Downloaded {model_name} to {destination}")
        return True

    except Exception as e:
        logger.error(f"Error downloading {model_name}: {e}")
        return False


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Dict with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def print_model_info(model: torch.nn.Module, name: str = "Model"):
    """
    Print model information

    Args:
        model: PyTorch model
        name: Model name for display
    """
    params = count_parameters(model)

    logger.info(f"{name} Info:")
    logger.info(f"  Total parameters: {params['total']:,}")
    logger.info(f"  Trainable parameters: {params['trainable']:,}")
    logger.info(f"  Frozen parameters: {params['frozen']:,}")


def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize model for inference

    Args:
        model: PyTorch model

    Returns:
        Optimized model
    """
    # Set to eval mode
    model.eval()

    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False

    # Enable inference optimizations
    if hasattr(torch, 'inference_mode'):
        logger.debug("Using torch.inference_mode optimizations")

    return model


def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get the device of a model

    Args:
        model: PyTorch model

    Returns:
        Device of model
    """
    return next(model.parameters()).device


def move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    Move a batch dictionary to device

    Args:
        batch: Batch dictionary
        device: Target device

    Returns:
        Batch on target device
    """
    moved_batch = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif isinstance(value, dict):
            moved_batch[key] = move_batch_to_device(value, device)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            moved_batch[key] = [v.to(device) for v in value]
        else:
            moved_batch[key] = value

    return moved_batch


class ModelCache:
    """
    Cache for loaded models to avoid reloading
    Singleton pattern
    """
    _instance = None
    _cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get(cls, model_name: str) -> Optional[torch.nn.Module]:
        """Get model from cache"""
        return cls._cache.get(model_name)

    @classmethod
    def set(cls, model_name: str, model: torch.nn.Module):
        """Add model to cache"""
        cls._cache[model_name] = model
        logger.debug(f"Cached model: {model_name}")

    @classmethod
    def clear(cls, model_name: Optional[str] = None):
        """Clear cache"""
        if model_name is None:
            cls._cache.clear()
            logger.debug("Cleared all cached models")
        elif model_name in cls._cache:
            del cls._cache[model_name]
            logger.debug(f"Cleared cached model: {model_name}")

    @classmethod
    def list_cached(cls) -> list:
        """List cached model names"""
        return list(cls._cache.keys())


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    step: int,
    loss: float,
    save_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save model checkpoint

    Args:
        model: Model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch
        step: Current step
        loss: Current loss
        save_path: Path to save checkpoint
        metadata: Additional metadata to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if metadata is not None:
        checkpoint['metadata'] = metadata

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)

    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load model checkpoint

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint on

    Returns:
        Checkpoint dictionary
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu')

    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model state from {checkpoint_path}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded optimizer state from {checkpoint_path}")

    return checkpoint


# Global model cache instance
model_cache = ModelCache()
