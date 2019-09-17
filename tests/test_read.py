import sys
import argparse
import time
import vidi
import numpy as np
import matplotlib.pyplot as plt


# using looped batch is faster than
"""
python test_read.py --frames 1000 --batch 100 --outtype "numpy" --debug
# 2629.947 ms
python test_read.py --frames 1000 --batch 100 --outtype "numpy" --asloop --debug
# 1618.628 ms

python test_read.py --frames 1000 --batch 10 --outtype "numpy" --asloop --debug
# 1500.711 ms


python test_read.py --frames 1000 --batch 10 --outtype "torch" --asloop --debug
# 1636.190 ms
"""

SRC = "/home/z/work/gans/pix2pix/results/metropolis_pix.mov"
FRAMES = 100
BATCH_SIZE = 10
DTYPE = "float32"
OUTTYPE = "numpy"
DEVICE = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=SRC, help='valid video file')
parser.add_argument('--frames', type=int, default=FRAMES, help='read number of frames')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch_size')
parser.add_argument('--dtype', type=str, default=DTYPE, help='dtype')
parser.add_argument('--outtype', type=str, default=OUTTYPE, help='numpy | torch')
parser.add_argument('--device', type=str, default=DEVICE, help='cpu | cuda')
parser.add_argument('--asloop', help='use loop read', action='store_true')
parser.add_argument('--debug', help='debug', action='store_true')
args = parser.parse_args()


def main(src, frames, batch_size, dtype, outtype, device, asloop, debug):

    _st = None
    _pf = "rgb24"
    _gr = False

    start = time.time()
    with vidi.FFread(src, batch_size=batch_size, out_type=outtype, start=_st, frames=frames,
                     pix_fmt=_pf, debug=debug, dtype=dtype, device=device, grad=_gr) as F:
        _data = np.zeros([batch_size, 480, 640, 3])
        while F.framecount < F.frames:
            if asloop:
                F.get_batch_loop()
            else:
                F.get_batch()
            # print(F.data.shape, F.data.max(), F.data.min())
            # _data = F.data.copy()
            #_data[:] = F.data[:]

    # plt.imshow(_data[0])
    # plt.show()
    print("TOTAL %.3fs"%(time.time()-start))



if __name__ == "__main__":
    main(src=args.src, frames=args.frames, batch_size=args.batch_size, dtype=args.dtype,
         outtype=args.outtype, device=args.device, asloop=args.asloop, debug=args.debug)
