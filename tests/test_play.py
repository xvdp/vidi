import sys
import os
import os.path as osp
import argparse
import vidi
from vidi.utils import Col

SRC = "metro%08d.png"
SRC = "*.jpg"
FOLDER = "/media/z/Elements/data/Colorization/train"
FRAMES = None

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=int, default=0, help='0, runs imgs test, 1 runs mov test, 2, runs test specified by srs and folder')
parser.add_argument('--src', type=str, default=SRC, help='video, ')
parser.add_argument('--folder', type=str, default=FOLDER, help='read number of frames')
parser.add_argument('--start_frame', type=int, default=0, help='read number of frames')
parser.add_argument('--start_time', type=float, default=0.0, help='read number of frames')
parser.add_argument('--fps', type=float, default=30, help='read number of frames')
parser.add_argument('--fullscreen', action='store_true')

args = parser.parse_args()

def main(test, src, folder, start, fps, fullscreen):
    if test == 0:
        run_imgs_test(start=start, fps=fps, fullscreen=fullscreen)
    elif test == 1:
        run_mov_test(start=start, fps=fps, fullscreen=fullscreen)
    else:
        vidi.ffplay(src, folder=folder, start=start, fps=fps, fullscreen=fullscreen)

def run_imgs_test(folder="/media/z/Elements/data/Colorization/train", src="*.jpg", start=None, fps=30, fullscreen=False):
    vidi.ffplay(src, folder=folder, start=start, fps=fps, fullscreen=fullscreen)

def run_mov_test(folder="/home/z/work/gans/pix2pix/results", src="metropolis_charlie_xavier_rnd.mov", start=None, fps=30, fullscreen=False):
    vidi.ffplay(src, folder=folder, start=start, fps=fps, fullscreen=fullscreen)

    

if __name__ == "__main__":
    _start = None
    if args.start_frame:
        _start = args.start_frame
    if args.start_time:
        if _start is not None:
            print(Col.YB, "overriding start frame %d with start time %f"%(args.start_frame, args.start_time))
        _start = args.start_time

    main(args.test, args.src, args.folder, _start, args.fps, args.fullscreen)

        



