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


class WrapBitTranscoder(Transcoder):
    """
    Encode depth images by taking the highest bits and
    packing them into the highest bits of an RGB image.

    Since red is often neglected in compression
    algorithms, the 8 MSBs are encoded into green and blue
    and the 8 LSBs are encoded into red.
    """

    def to_rgb(self, depth_image):
        bits = []
        ones = np.ones_like(depth_image)
        for i in range(0, 16):
            bits.append(((depth_image & (ones << i)) != 0).astype('uint8'))
        image = np.zeros(depth_image.shape + (3,), dtype='uint8')
        for i in range(8, 16):
            image[..., 1 + (i % 2)] |= bits[i] << (4 + ((i - 8) // 2))
        for i in range(0, 8):
            image[..., 0] |= bits[i] << i
        return image

    def to_depth(self, rgb_image):
        bits = []
        ones = np.ones_like(rgb_image[..., 0])
        for i in range(0, 8):
            bits.append((rgb_image[..., 0] & (ones << i)) != 0)
        for i in range(8, 16):
            bits.append((rgb_image[..., 1 + (i % 2)] & (ones << (4 + ((i - 8) // 2)))) != 0)
        result = np.zeros(rgb_image.shape[:-1], dtype='uint16')
        for i, b in enumerate(bits):
            result |= b.astype('uint16') << i
        return result
