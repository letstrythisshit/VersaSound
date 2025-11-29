"""
Device management utilities for VersaSound
Handles automatic device detection, memory management, and optimization
"""

import torch
import logging
from typing import Optional, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages device selection and memory optimization
    Automatically detects best available device (CUDA, MPS, CPU)
    """

    _instance = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._device is None:
            self._device = self._detect_device()
            logger.info(f"Initialized DeviceManager with device: {self._device}")

    @staticmethod
    def _detect_device() -> torch.device:
        """Detect best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon) device")
            return torch.device('mps')
        else:
            logger.warning("No GPU available, using CPU (this will be slow)")
            return torch.device('cpu')

    @property
    def device(self) -> torch.device:
        """Get current device"""
        return self._device

    def set_device(self, device: Union[str, torch.device]):
        """Manually set device"""
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        logger.info(f"Device manually set to: {device}")

    def get_memory_info(self) -> dict:
        """Get memory information for current device"""
        if self._device.type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
                'total': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'unit': 'GB'
            }
        else:
            return {'message': 'Memory info only available for CUDA devices'}

    def clear_cache(self):
        """Clear GPU cache"""
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared CUDA cache")

    def optimize_memory(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply memory optimizations to model

        Args:
            model: PyTorch model to optimize

        Returns:
            Optimized model
        """
        # Move to device
        model = model.to(self._device)

        # Enable gradient checkpointing if available
        if hasattr(model, 'enable_gradient_checkpointing'):
            try:
                model.enable_gradient_checkpointing()
                logger.debug("Enabled gradient checkpointing")
            except Exception as e:
                logger.debug(f"Could not enable gradient checkpointing: {e}")

        # Set to eval mode for inference
        model.eval()

        return model

    @contextmanager
    def autocast_context(self, enabled: bool = True, dtype: torch.dtype = torch.float16):
        """
        Context manager for automatic mixed precision

        Args:
            enabled: Whether to enable autocast
            dtype: Data type for autocast (float16 or bfloat16)
        """
        if self._device.type == 'cuda' and enabled:
            with torch.cuda.amp.autocast(dtype=dtype):
                yield
        else:
            yield

    def get_optimal_batch_size(self, base_size: int = 8) -> int:
        """
        Get optimal batch size based on available memory

        Args:
            base_size: Base batch size

        Returns:
            Adjusted batch size
        """
        if self._device.type != 'cuda':
            return base_size

        mem_info = self.get_memory_info()
        available_memory = mem_info['total'] - mem_info['allocated']

        # Heuristic: reduce batch size if memory is limited
        if available_memory < 4.0:  # Less than 4GB available
            return max(1, base_size // 2)
        elif available_memory < 8.0:  # Less than 8GB available
            return base_size
        else:  # More than 8GB available
            return base_size * 2

    def __repr__(self) -> str:
        return f"DeviceManager(device={self._device})"


# Global device manager instance
_device_manager = DeviceManager()


def get_device() -> torch.device:
    """Get current device"""
    return _device_manager.device


def set_device(device: Union[str, torch.device]):
    """Set device"""
    _device_manager.set_device(device)


def clear_cache():
    """Clear GPU cache"""
    _device_manager.clear_cache()


def get_memory_info() -> dict:
    """Get memory information"""
    return _device_manager.get_memory_info()


def optimize_memory(model: torch.nn.Module) -> torch.nn.Module:
    """Optimize model memory usage"""
    return _device_manager.optimize_memory(model)


@contextmanager
def autocast(enabled: bool = True, dtype: torch.dtype = torch.float16):
    """Autocast context manager"""
    with _device_manager.autocast_context(enabled, dtype):
        yield
