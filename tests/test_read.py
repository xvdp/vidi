import sys
import argparse
import time
import vidi
import numpy as np
import matplotlib.pyplot as plt



SRC = "/home/z/work/gans/pix2pix/results/metropolis_pix.mov"
FRAMES = 100
BATCH_SIZE = 10
DTYPE = "float32"
OUTTYPE = "numpy"

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=SRC, help='valid video file')
parser.add_argument('--frames', type=int, default=FRAMES, help='read number of frames')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch_size')
parser.add_argument('--dtype', type=str, default=DTYPE, help='dtype')
parser.add_argument('--outtype', type=str, default=OUTTYPE, help='numpy | torch')
parser.add_argument('--debug', help='debug', action='store_true')
args = parser.parse_args()


def main(src, frames, batch_size, dtype, outtype, debug):

    _st = None
    _pf = "rgb24"
    _dv = "cpu"
    _gr = False

    start = time.time()
    with vidi.FFread(src, batch_size=batch_size, out_type=outtype, start=_st, frames=frames, pix_fmt=_pf, debug=debug, dtype=dtype, device=_dv, grad=_gr) as F:
        _data = np.zeros([batch_size, 480, 640, 3])
        while F.framecount < F.frames:
            F.get_batch()
            # print(F.data.shape, F.data.max(), F.data.min())
            # _data = F.data.copy()
            #_data[:] = F.data[:]

    # plt.imshow(_data[0])
    # plt.show()
    print("TOTAL %.3fs"%(time.time()-start))




if __name__ == "__main__":
    main(src=args.src, frames=args.frames, batch_size=args.batch_size, dtype=args.dtype, outtype=args.outtype, debug=args.debug)