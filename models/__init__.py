"""
Standalone Models Package

Import models directly:
    from models import DeltaNet, Transformer

Each model is a pure PyTorch implementation using FLA kernels.
"""

from .deltanet import DeltaNet
from .transformer import Transformer

__all__ = [
    'DeltaNet',
    'Transformer',
]

__version__ = '0.1.0'
