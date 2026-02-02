"""
Seed management for reproducibility.

Provides utilities to set deterministic seeds across all random number generators
used in the pipeline (Python, NumPy, PyTorch).
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
        deterministic: If True, enables PyTorch deterministic algorithms.
                      Note: This may impact performance.
    
    Example:
        >>> set_seed(42)
        >>> # All subsequent random operations will be reproducible
    """
    # Python random
    random.seed(seed)
    
    # Environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    if deterministic:
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # PyTorch 1.8+ deterministic flag
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass  # Some operations may not have deterministic implementations
    else:
        # Enable cuDNN auto-tuner for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_worker_seed_fn(seed: int):
    """
    Get a worker initialization function for DataLoader.
    
    This ensures each worker has a unique but reproducible seed based on
    the base seed and worker ID.
    
    Args:
        seed: Base random seed.
    
    Returns:
        Worker initialization function.
    
    Example:
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=get_worker_seed_fn(42)
        ... )
    """
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn


class SeedContext:
    """
    Context manager for temporarily setting a specific seed.
    
    Useful for operations that need to be reproducible while not affecting
    the global random state.
    
    Example:
        >>> with SeedContext(123):
        ...     # Operations here use seed 123
        ...     x = torch.randn(10)
        >>> # Random state is restored after the context
    """
    
    def __init__(self, seed: int):
        """
        Initialize seed context.
        
        Args:
            seed: Random seed to use within the context.
        """
        self.seed = seed
        self._python_state: Optional[object] = None
        self._numpy_state: Optional[dict] = None
        self._torch_state: Optional[torch.Tensor] = None
        self._cuda_states: list = []
    
    def __enter__(self) -> "SeedContext":
        """Save current random states and set new seed."""
        # Save states
        self._python_state = random.getstate()
        self._numpy_state = np.random.get_state()
        self._torch_state = torch.get_rng_state()
        
        if torch.cuda.is_available():
            self._cuda_states = [
                torch.cuda.get_rng_state(device=i)
                for i in range(torch.cuda.device_count())
            ]
        
        # Set new seed
        set_seed(self.seed, deterministic=False)
        
        return self
    
    def __exit__(self, *args) -> None:
        """Restore previous random states."""
        if self._python_state is not None:
            random.setstate(self._python_state)
        
        if self._numpy_state is not None:
            np.random.set_state(self._numpy_state)
        
        if self._torch_state is not None:
            torch.set_rng_state(self._torch_state)
        
        if torch.cuda.is_available() and self._cuda_states:
            for i, state in enumerate(self._cuda_states):
                torch.cuda.set_rng_state(state, device=i)


def generate_fold_seeds(base_seed: int, n_folds: int) -> list:
    """
    Generate unique seeds for each fold in cross-validation.
    
    Args:
        base_seed: Base random seed.
        n_folds: Number of folds.
    
    Returns:
        List of seeds, one per fold.
    """
    rng = np.random.RandomState(base_seed)
    return rng.randint(0, 2**31 - 1, size=n_folds).tolist()
