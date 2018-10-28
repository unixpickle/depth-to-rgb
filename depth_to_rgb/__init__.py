"""
Conversion between 16-bit depth images and 24-bit RGB
images.
"""

from .base import Transcoder
from .benchmark import benchmark_transcoder, print_benchmark_results, transcode_at_quality
from .naive import GrayscaleTranscoder, WrapBitTranscoder
from .halftone import HalfToneCheatingTranscoder

__all__ = ['GrayscaleTranscoder', 'Transcoder', 'WrapBitTranscoder', 'benchmark_transcoder',
           'print_benchmark_results', 'HalfToneCheatingTranscoder', 'transcode_at_quality']
