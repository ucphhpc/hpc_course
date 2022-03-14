#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib.pyplot import imshow, draw, pause
import matplotlib.pyplot as plt
from matplotlib import animation
import h5py
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
        plt_im = plt.imshow(image, cmap='coolwarm', animated=True)
        plt_txt = plt.text(10, 310, label, color='red')
        self.images.append([plt_im, plt_txt])

    def save(self, filename, fps):
        ani = animation.ArtistAnimation(self.fig, self.images)
        ani.save(filename, writer='imagemagick', fps=fps)


def main(args):
    with h5py.File(args.input_hdf5_file, 'r') as f:
        video = AnimatedGif()
        for frame_id in sorted([int(k) for k in f.keys()]):
            frame = np.array(f[str(frame_id)]['/%d/world' % frame_id])
            video.add(frame)
        video.save(args.output_video_file, args.fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the n-body NICE model')
    parser.add_argument(
        'input_hdf5_file',
        type=str,
        help='Path to the hdf5 file to visualize.'
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
        '--format',
        choices=["gif", "mp4"],
        type=str,
        default="gif",
        help='Frames per second.'
    )
    args = parser.parse_args()
    main(args)
