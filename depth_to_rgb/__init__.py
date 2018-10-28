"""
Conversion between 16-bit depth images and 24-bit RGB
images.
"""

from .base import Transcoder
from .benchmark import benchmark_transcoder, print_benchmark_results
from .naive import GrayscaleTranscoder, WrapBitTranscoder, HalfToneCheatingTranscoder

__all__ = ['GrayscaleTranscoder', 'Transcoder', 'WrapBitTranscoder',
           'HalfToneCheatingTranscoder', 'benchmark_transcoder', 'print_benchmark_results']
