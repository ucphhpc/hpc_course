#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, pyplot as plt
import h5py
import argparse
import numpy as np


def plot_surface_video(time_steps, x, y, filename, fps, figsize=(100, 100)):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, time_steps[0])

    def animate(time_step):
        print('hej')
        ax.clear()
        ax.set_zlim((0, time_steps[0].max() * 2))
        line = ax.plot_surface(x, y, time_step)
        return (line,)

    ani = animation.FuncAnimation(fig, animate, time_steps, interval=50, blit=True)
    print("hej")
    ani.save(filename, writer='imagemagick', fps=fps)


def main(args):
    with h5py.File(args.input_hdf5_file, 'r') as f:
        time_steps = []
        for frame_id in sorted([int(k) for k in f.keys()]):
            frame = np.array(f[str(frame_id)]['/%d/water' % frame_id])
            time_steps.append(frame)
        shape = time_steps[0].shape
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        plot_surface_video(time_steps, x, y, args.output_video_file, args.fps)


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
