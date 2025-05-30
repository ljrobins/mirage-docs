"""
Animating Subframes
===================
"""

import os
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mirage.photo import background_subtract, load_fits, plot_fits

ds = [
    '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/pogs/2025/2025_02_25/echostar 2 1s/*',
    '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/pogs/2025/2025_03_01/delta-1-3s/*',
]

target_fps = 7

for d in ds:
    fits_files = glob(d)
    fits_files = sorted(fits_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    if 'echo' in d:
        fits_files = fits_files[::7]

    obj_name = fits_files[0].split('/')[-1].split('.')[1][2:-1].replace('_', ' ')

    fig = plt.figure(figsize=(5, 5))
    fig.patch.set_facecolor('w')

    def animate(i: int) -> None:
        print(i)
        plt.cla()
        plt.title(f'{obj_name} ({i}/{len(fits_files)})', color='k')
        return [plot_fits(background_subtract(load_fits(fits_files[i])[1])[0])]

    animate(0)
    plt.tight_layout()

    interval = 1000 / target_fps

    anim = FuncAnimation(
        plt.gcf(), animate, frames=len(fits_files), interval=interval, blit=True
    )
    anim.save(
        os.path.join(
            '/Users/liamrobinson/Documents/maintained-research/mirage/examples/15-attitude-sdc-2025/out',
            f'{obj_name}.mp4',
        ),
        writer='ffmpeg',
        fps=target_fps,
    )

    plt.show()
