"""
A collection of obviously naive algorithms.
"""

import numpy as np

from .base import Transcoder


class GrayscaleTranscoder(Transcoder):
    """
    Encode depth images as grayscale.
    """

    def to_rgb(self, depth_image):
        return (np.stack([depth_image] * 3, axis=-1) >> 8).astype('uint8')

    def to_depth(self, rgb_image):
        rgb_image = rgb_image.astype('uint16')
        return ((rgb_image[..., 0] + rgb_image[..., 1] + rgb_image[..., 2]) // 3) << 8
