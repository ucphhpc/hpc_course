#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, pyplot as plt
import h5py
import argparse
import numpy as np


class AnimatedGif:
    def __init__(self, zlim, size=(800, 800)):
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / 100, size[1] / 100)
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_zlim(zlim)
        self.images = []

    def add(self, image):
        x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
        plt_im = self.ax.plot_surface(x, y, image, color='b')
        self.images.append([plt_im])

    def save(self, filename, fps):
        ani = animation.ArtistAnimation(self.fig, self.images)
        ani.save(filename, writer='imagemagick', fps=fps)


def main(args):
    with h5py.File(args.input_hdf5_file, 'r') as f:
        video = AnimatedGif(zlim=(0, args.zlim))
        count = 0
        for frame_id in sorted([int(k) for k in f.keys()]):
            if frame_id >= args.start:
                frame = np.array(f[str(frame_id)]['/%d/water' % frame_id])
                if count % args.step == 0:
                    video.add(frame)
                count += 1
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
        '--step',
        type=int,
        default=10,
        help='Frames to skip to get to the next frame (default is 10).'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start frame number (default is 0, which is the first frame).'
    )
    parser.add_argument(
        '--zlim',
        type=int,
        default=2,
        help='Set the max elevation.'
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
