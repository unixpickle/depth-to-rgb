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
        compression levels to MSEs.
    """
    result_dict = {}
    for bench_name, image in load_depth_images():
        bench_dict = {}
        for quality in range(10, 101, 10):
            data = io.BytesIO()
            rgb = transcoder.to_rgb(image)
            assert rgb.dtype == 'uint8'
            Image.fromarray(rgb).save(data, format='JPEG', quality=quality)
            decoded = transcoder.to_depth(np.array(Image.open(data), 'uint8'))
            assert decoded.dtype == 'uint16'
            error = np.mean(np.square((decoded.astype('float') - image.astype('float')) / 0xffff))
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
            print('  %d - %.4e' % (quality, error))


def load_depth_images():
    """
    Load testing depth images.

    Returns:
      An iterator over (name, image_array) tuples.
    """
    asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    if not os.path.exists(asset_dir):
        raise FileNotFoundError('missing asset directory at: ' + asset_dir)
    for name in os.listdir(asset_dir):
        if name.endswith('.png'):
            yield (name, np.array(Image.open(os.path.join(asset_dir, name)), 'uint16'))
