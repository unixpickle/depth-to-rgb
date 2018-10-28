"""
Compare a depth image before and after transcoding.
"""

import argparse
import matplotlib.pyplot as plt

import depth_to_rgb

TRANSCODERS = {
    'grayscale': depth_to_rgb.GrayscaleTranscoder(),
    'wrapbit': depth_to_rgb.WrapBitTranscoder(),
    'halftone-cheat': depth_to_rgb.HalfToneCheatingTranscoder(),
}


def main(args):
    for name, coder in TRANSCODERS.items():
        print('Benchmarking {}...'.format(name))
        results = depth_to_rgb.transcode_at_quality(coder, args.quality)
        fig, axes = plt.subplots(2, len(results))
        for i, (bench_name, bench_dict) in enumerate(sorted(results.items())):
            axes[0, i].imshow(bench_dict['original'])
            axes[1, i].imshow(bench_dict['decoded'])
            axes[0, i].set_title(bench_name, fontsize=8)
        for ax, row in zip(axes[:, 0], ['original', 'decoded']):
            ax.set_ylabel(row, fontsize=8)
        for ax in axes.flatten():
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        plt.show()


if __name__ == '__main__':
    def str2bool(s):
        return s.lower() in ('true')

    parser = argparse.ArgumentParser(description='Transcoder Visualization')
    parser.add_argument('--quality', type=int, default=50,
                        help='Control the quality of the compression. The higher the less lossy.')
    args, unparsed = parser.parse_known_args()
    main(args)
