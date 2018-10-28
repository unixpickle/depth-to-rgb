"""
Measure the performance of a Transcoder.
"""

import io
import os

from PIL import Image
import numpy as np


def benchmark_transcoder(transcoder):
    """
    Evaluate the performance of a transcoder.

    Args:
      transcoder: the Transcoder to benchmark.

    Returns:
      A dict mapping benchmarks to dicts mapping
        compression levels to mean absolute errors.
    """
    result_dict = {}
    for bench_name, image in load_depth_images():
        bench_dict = {}
        for quality in range(10, 101, 10):
            decoded = compression_reconstruction(transcoder, image, quality)
            error = np.mean(np.abs(decoded.astype('float') - image.astype('float')))
            bench_dict[quality] = error
        result_dict[bench_name] = bench_dict
    return result_dict


def print_benchmark_results(results):
    """
    Print the results of benchmark_transcoder in a
    human-readable format.
    """
    for bench_name, bench_dict in sorted(results.items()):
        print('Results for ' + bench_name)
        for quality, error in sorted(bench_dict.items()):
            print('  %d - %f' % (quality, error))


def transcode_at_quality(transcoder, quality):
    """Transcode and compare a depth image at a fixed compression level.

    Args:
      transcoder: the Transcoder to use.
      quality: the compression level.

    Returns:
      A dict mapping a transcoder to a dict containing
      the original image and its decoded equivalent.
    """
    result_dict = {}
    for bench_name, image in load_depth_images():
        bench_dict = {}
        bench_dict['original'] = image
        bench_dict['decoded'] = compression_reconstruction(transcoder, image, quality)
        result_dict[bench_name] = bench_dict
    return result_dict


def load_depth_images():
    """
    Load testing depth images.

    Returns:
      An iterator over (name, image_array) tuples.
    """
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images')
    if not os.path.exists(image_dir):
        raise FileNotFoundError('missing image directory at: ' + image_dir)
    for name in os.listdir(image_dir):
        if name.endswith('.png'):
            yield (name, np.array(Image.open(os.path.join(image_dir, name)), 'uint16'))


def compression_reconstruction(transcoder, image, quality):
    """
    Get a reconstruction after encoding, compressing, and
    decoding an image.
    """
    data = io.BytesIO()
    rgb = transcoder.to_rgb(image)
    assert rgb.dtype == 'uint8'
    Image.fromarray(rgb).save(data, format='JPEG', quality=quality)
    decoded = transcoder.to_depth(np.array(Image.open(data), 'uint8'))
    assert decoded.dtype == 'uint16'
    return decoded
