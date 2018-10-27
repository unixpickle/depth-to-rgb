"""
Conversion between 16-bit depth images and 24-bit RGB
images.
"""

from .base import Transcoder
from .benchmark import benchmark_transcoder, print_benchmark_results
from .naive import GrayscaleTranscoder

__all__ = ['GrayscaleTranscoder', 'Transcoder', 'benchmark_transcoder', 'print_benchmark_results']
