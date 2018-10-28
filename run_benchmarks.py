"""
Benchmark all of the algorithms.
"""

import numpy as np

import depth_to_rgb

TRANSCODERS = {
    'grayscale': depth_to_rgb.GrayscaleTranscoder(),
    'wrapbit': depth_to_rgb.WrapBitTranscoder(),
}


def main():
    all_results = {}
    for name, coder in TRANSCODERS.items():
        print('Benchmarking %s...' % name)
        results = depth_to_rgb.benchmark_transcoder(coder)
        depth_to_rgb.print_benchmark_results(results)
        all_results[name] = results
    save_markdown_results('RESULTS.md', all_results)


def save_markdown_results(filename, all_results):
    output_data = ('| Name | 10% quality | 50% quality | 100% quality |\n' +
                   '|:----:|:-----------:|:-----------:|:------------:|\n')
    for name, results in all_results.items():
        def _mean(quality):
            return np.mean([x[quality] for x in results.values()])
        output_data += ('| %s | %.2f | %.2f | %.2f |\n' %
                        (name, _mean(10), _mean(50), _mean(100)))
    with open(filename, 'wt+') as out_file:
        out_file.write(output_data)


if __name__ == '__main__':
    main()
