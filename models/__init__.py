"""
Consumer GPU Video Generation - Model Wrappers
Optimized wrappers for video generation models with INT4 quantization support.
"""

from .wan22 import Wan22Pipeline
from .ltx2 import LTX2Pipeline

__all__ = ["Wan22Pipeline", "LTX2Pipeline"]
