import argparse
import os.path as osp
import numpy as np
from PIL import Image
import vidi

"""

"""

parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int, default=0, help='0: test_cap, 1: test_with_cap')
parser.add_argument('--name', type=str, default='vidi_test', help='file name')
parser.add_argument('--ext', type=str, default='.avi', help='.mov, .avi, .mp4, .flv')
parser.add_argument('--pix_fmt', type=str, default='rgb24', help='rgb24, gray, yuv420p')
parser.add_argument('--img', type=str, default='/home/z/data/bundaberg_bully.jpg', help='valid image file')
parser.add_argument('--crash', type=int, default=0, help='test crash to see if it works')
args = parser.parse_args()



def get_img(img_path):
    assert osp.isfile(img_path), "file not found, enter different valid filename"
    return np.array(Image.open(img_path))

def test_cap(name, ext, pix_fmt, img_path, crash=0):

    img = get_img(img_path)
    size = img.shape[:2]

    F = vidi.FFcap(name+ext, pix_fmt=pix_fmt, size=size, overwrite=True, debug=True)
    F.from_frames()

    c = img.shape[2]
    if pix_fmt == "gray":
        c = 1
        img = img[:, :, 1:2]
    zero = np.zeros([size[0], size[1], c], dtype=np.uint8)
    one = np.ones([size[0], size[1], c], dtype=np.uint8)*255

    for i in range(20):
        if crash and i == 19:
            assert False, "break to ensure close works"
        for j in range(10):
            F.add_frame(img)
        for j in range(10):
            F.add_frame(one)
        for j in range(10):
            F.add_frame(zero)
        for j in range(10):
            F.add_frame((img*0.5).astype(np.uint8))
    F.close()


def test_with(name, ext, pix_fmt, img_path, crash=0):
    name = "with_"+name
    img = get_img(img_path)
    size = img.shape[:2]

    with vidi.FFcap(name+ext, pix_fmt=pix_fmt, size=size, overwrite=True, debug=True) as F:
        F.from_frames()

        c = img.shape[2]
        if pix_fmt == "gray":
            c = 1
            img = img[:, :, 1:2]
        zero = np.zeros([size[0], size[1], c], dtype=np.uint8)
        one = np.ones([size[0], size[1], c], dtype=np.uint8)*255
        print(one.shape)

        for i in range(20):
            if crash and i == 19:
                assert False, "break to ensure close works"
            for j in range(10):
                F.add_frame(img)
            for j in range(10):
                F.add_frame(one)
            for j in range(10):
                F.add_frame(zero)
            for j in range(10):
                F.add_frame((img*0.5).astype(np.uint8))

if __name__ == "__main__":
    # test_cap("cap_avi", ".avi" 'rgb24', args.img)
    # test_cap("cap_mov_raw_gray", ".mov", 'raw', 'gray', args.img)

    _num = int(args.num)

    if not _num:
        test_cap(args.name, args.ext, args.pix_fmt, args.img, args.crash)
    else:
        test_with(args.name, args.ext, args.pix_fmt, args.img, args.crash)

