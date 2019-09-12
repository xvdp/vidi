import argparse
import os.path as osp
import numpy as np
from PIL import Image
import vidi


parser = argparse.ArgumentParser()
parser.add_argument('--ext', type=str, default='.avi', help='.mov, .avi, .mp4, .flv')
parser.add_argument('--codec', type=str, default='raw', help='raw, h264')
parser.add_argument('--pix_fmt', type=str, default='rgb24', help='rgb24, gray, yuv420p')
parser.add_argument('--img', type=str, default='/home/z/data/bundaberg_bully.jpg', help='valid image file')
args = parser.parse_args()


def get_img(img_path):
    assert osp.isfile(img_path), "file not found, enter different valid filename"
    return np.array(Image.open(img_path))



def test_cap(ext, codec, pix_fmt, img_path):

    img = get_img(img_path)
    size = img.shape[:2]

    F = vidi.FFcap("vidi_test"+ext, codec=codec, pix_fmt=pix_fmt, size=size, overwrite=True, debug=True)
    F.from_frames()

    c = img.shape[2]
    if pix_fmt == "gray":
        c = 1
        img = img[:,:,1:2]
    zero = np.zeros([size[0], size[1], c], dtype=np.uint8)
    one = np.ones([size[0], size[1], c], dtype=np.uint8)*255
    print(one.shape)

    for i in range(20):
        for j in range(10):
            F.add_frame(img)
        for j in range(10):
            F.add_frame(one)
        for j in range(10):
            F.add_frame(zero)
        for j in range(10):
            F.add_frame((img*0.5).astype(np.uint8))
    F.close()

if __name__ == "__main__":
    test_cap(args.ext, args.codec, args.pix_fmt, args.img)
