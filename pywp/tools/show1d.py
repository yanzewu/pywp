
from .. import visualize, snapshot

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Display 1D wavepacket simulation data')

    parser.add_argument('-a', '--vmax', type=float, help='relative vmax', default=2.0)
    parser.add_argument('-m' , '--momentum', action='store_const', help='display momentum', default=False, const=True)
    parser.add_argument('-c', '--coord', type=int, help='Coordinate to display', default=0)
    parser.add_argument('--figsize', type=lambda x:tuple(map(float, x.split(','))), help='size of figure', default=(6.4, 4.8))
    parser.add_argument('--fps', type=int, help='fps of video', default=5)
    parser.add_argument('--dpi', type=int, help='dpi of video', default=192)
    parser.add_argument('--video', help='filename of video', default='')
    parser.add_argument('--autoplay', action='store_const', help='auto play video', default=False, const=True)
    parser.add_argument('filename', help='filename')

    args = parser.parse_args()
    
    s = snapshot.load_file(args.filename)
    if not args.video:
        visualize.visualize_snapshot_1d(s, axis=args.coord, relative_vmax=args.vmax, show_momentum=args.momentum, interactive=not args.autoplay, figsize=args.figsize)
    else:
        outputs = visualize.visualize_snapshot_1d(s, axis=args.coord, relative_vmax=args.vmax, show_momentum=args.momentum, figsize=args.figsize, interactive=False, output_callback=True)
        visualize.write_video(args.video, *outputs, fps=args.fps, dpi=args.dpi)
