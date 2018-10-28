"""
A collection of halftone transcoders.
"""

import numpy as np

from .base import Transcoder


class HalfToneCheatingTranscoder(Transcoder):
    """
    Encode depth images by normalizing the values to [0, 1] range.

    Instead of dividing by the theoretical max value of 65535 (2^16 -1),
    we do min-max normalization to better scale the values. Note
    that this version cheats by hardcoding the max depth value.
    """

    def _halftone(self, image, min_val, max_val):
        image = np.copy(image).astype('float64')
        image -= min_val
        image *= (255 / (max_val - min_val))
        return np.floor(image).astype('uint8')

    def to_rgb(self, depth_image):
        lut = np.arange(2**16, dtype=np.uint16)
        lut = self._halftone(lut, 0, 1140)
        depth_image_uint8 = np.take(lut, depth_image)
        return np.stack([depth_image_uint8] * 3, axis=-1)

    def to_depth(self, rgb_image):
        rgb_image = rgb_image.astype('float64')
        depth_image = ((rgb_image[..., 0] + rgb_image[..., 1] + rgb_image[..., 2]) // 3)
        depth_image = depth_image * (1140 - 0) / 255
        depth_image = depth_image + 0
        return np.round(depth_image).astype('uint16')
