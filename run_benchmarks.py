"""
Benchmark all of the algorithms.
"""

import depth_to_rgb

TRANSCODERS = {
    'grayscale': depth_to_rgb.GrayscaleTranscoder(),
}


def main():
    for name, coder in TRANSCODERS.items():
        print('Benchmarking %s...' % name)
        results = depth_to_rgb.benchmark_transcoder(coder)
        depth_to_rgb.print_benchmark_results(results)


if __name__ == '__main__':
    main()
