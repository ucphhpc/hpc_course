#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, pyplot as plt
import matplotlib.cm as cm
import argparse
import numpy as np


class AnimatedGif:
    def __init__(self, size=(800, 600)):
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / 100, size[1] / 100)
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.images = []

    def add(self, image, label=''):
        plt_im = plt.imshow(image, cmap=cm.Greys_r, animated=True)
        plt_txt = plt.text(10, 310, label, color='red')
        self.images.append([plt_im, plt_txt])

    def save(self, filename, fps):
        ani = animation.ArtistAnimation(self.fig, self.images)
        ani.save(filename, writer='imagemagick', fps=fps)


def main(args):
    video = AnimatedGif()
    data = np.fromfile(args.input_bin_file, dtype=np.float32)
    size = np.cbrt(data.size)
    assert size**3 == data.size
    size = size.astype(int)
    data = data.reshape([size.astype(int)] * 3)
    for frame_id in range(size):
        video.add(data[frame_id])
    video.save(args.output_video_file, args.fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the n-body NICE model')
    parser.add_argument(
        'input_bin_file',
        type=str,
        help='Path to the binary file to visualize.'
    )
    parser.add_argument(
        'output_video_file',
        type=str,
        help='Path to the written video file.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second.'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=1,
        help='Frames to skip to get to the next frame (default is 10).'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start frame number (default is 0, which is the first frame).'
    )
    parser.add_argument(
        '--format',
        choices=["gif", "mp4"],
        type=str,
        default="gif",
        help='Frames per second.'
    )
    args = parser.parse_args()
    main(args)
