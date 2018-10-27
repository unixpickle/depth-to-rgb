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


class HalfToneCheatTranscoder(Transcoder):
    """
    Encode depth images by smartly normalizing to the range [0, 1].

    This version cheats by hardcoding the max depth value.
    """

    def _halftone(self, image, min_val, max_val):
        image = np.copy(image).astype('float64')
        image -= min_val
        image *=  (255 / (max_val - min_val))
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


class HalfToneTranscoder(Transcoder):
    """
    Encode depth images by smartly normalizing to the range [0, 1].

    Storing the max in the top corner pixel isn't working right now
    since I'm naively splitting the 16 bits into 2 8-bit numbers
    which means any change at the bit level can blow up once
    I reconvert to an integer. Need to rethink this through.
    """

    def _halftone(self, image, min_val, max_val):
        image = np.copy(image).astype('float64')
        image -= min_val
        image *=  (255 / (max_val - min_val))
        return np.floor(image).astype('uint8')

    def to_rgb(self, depth_image):
        depth_max = depth_image.max()
        lut = np.arange(2**16, dtype=np.uint16)
        lut = self._halftone(lut, 0, depth_max)
        depth_image_uint8 = np.take(lut, depth_image)
        rgb_image = np.stack([depth_image_uint8] * 3, axis=-1)
        depth_max_bin = bin(depth_max)[2:].zfill(16)
        left, right = int(depth_max_bin[:8], 2), int(depth_max_bin[8:], 2)
        rgb_image[0, 0, 0] = left; rgb_image[0, 0, 1] = right
        return rgb_image

    def to_depth(self, rgb_image):
        left = rgb_image[0, 0, 0]
        right = rgb_image[0, 0, 1]
        depth_max_bin = bin(left)[2:].zfill(8) + bin(right)[2:].zfill(8)
        depth_max = int(depth_max_bin, 2)
        rgb_image = rgb_image.astype('float64')
        depth_image = ((rgb_image[..., 0] + rgb_image[..., 1] + rgb_image[..., 2]) // 3)
        depth_image = depth_image * depth_max / 255
        return np.round(depth_image).astype('uint16')