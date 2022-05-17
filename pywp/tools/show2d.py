
from .. import visualize, snapshot

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Display 2D wavepacket simulation data')

    parser.add_argument('-a', '--vmax', type=float, help='relative vmax', default=0.2)
    parser.add_argument('-m' , '--momentum', type=lambda x:x.lower() == 'true', help='display momentum', default=0, nargs='?', const=False)
    parser.add_argument('--figsize', type=lambda x:tuple(map(float, x.split(','))), help='size of figure', default=(6.4, 4.8))
    parser.add_argument('--fps', type=int, help='fps of video', default=5)
    parser.add_argument('--dpi', type=int, help='dpi of video', default=192)
    parser.add_argument('--video', help='filename of video', default='')
    parser.add_argument('filename', help='filename')

    args = parser.parse_args()

    s = snapshot.load_file(args.filename)
    if not args.video:
        visualize.visualize_snapshot_2d(s, relative_vmax=args.vmax, show_momentum=args.momentum, figsize=args.figsize)
    else:
        outputs = visualize.visualize_snapshot_2d(s, relative_vmax=args.vmax, show_momentum=args.momentum, figsize=args.figsize, interactive=False, output_callback=True)
        visualize.write_video(args.video, *outputs, fps=args.fps, dpi=args.dpi)
