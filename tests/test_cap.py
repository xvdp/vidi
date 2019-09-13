import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image
import vidi
from vidi import utils

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

def clean_test(name):
    if osp.isfile(name):
        print("file %s exists, removing"%name)
        os.remove(name)

def record(F, img, crash, size, pix_fmt):
    if pix_fmt == "gray":
        c = 1
        img = img[:, :, 1:2]

    c = img.shape[2]
    zero = np.zeros([size[0], size[1], c], dtype=np.uint8)
    one = np.ones([size[0], size[1], c], dtype=np.uint8)*255

    for i in range(4):
        if crash and i == 3:
            assert False, "break to ensure close works"
        for j in range(10):
            F.add_frame(img)
        for j in range(10):
            F.add_frame(one)
        for j in range(10):
            F.add_frame(zero)
        for j in range(10):
            _img = img.copy().astype(float)
            _img[:, :, i%c] *= np.random.random()
            F.add_frame((_img).astype(np.uint8))

def test_with(name, ext, pix_fmt, img_path, crash=0):
    name = "with_"+name+ext
    img = get_img(img_path)
    clean_test(name)
    size = img.shape[:2]

    with vidi.FFcap(name, pix_fmt=pix_fmt, size=size, overwrite=True, debug=True) as F:
        record(F, img, crash, size, pix_fmt)

    return name

def test_cap(name, ext, pix_fmt, img_path, crash=0):

    img = get_img(img_path)
    name = name+ext
    clean_test(name)
    size = img.shape[:2]

    F = vidi.FFcap(name, pix_fmt=pix_fmt, size=size, overwrite=True, debug=True)
    F.open()
    record(F, img, crash, size, pix_fmt)
    F.close()
    return name

def run_tests():
    _fail = []
    _f = utils.Col.RB
    _s = utils.Col.GB
    _n = utils.Col.AU
    _t = 0
    try:
        cap = test_cap(args.name, args.ext, args.pix_fmt, args.img, args.crash)
        vidi.ffplay(cap, loop=1, autoexit=True, fullscreen=False)
        print("%sTest [%d], record, success\n---------------%s"%(_s,_t,_n))
    except:
        _fail.append(1)
        print("%sTest [%d], record, failure\n---------------%s"%(_f,_t,_n))
    _t +=1
    
    try:
        wcap = test_with(args.name, args.ext, args.pix_fmt, args.img, args.crash)
        vidi.ffplay(wcap, loop=1, autoexit=True, fullscreen=False)
        print("%sTest [%d], record, success\n---------------%s"%(_s,_t,_n))
    except:
        _fail.append(1)
        print("%sTest [%d], record, failure\n---------------%s"%(_f,_t,_n))
    _t += 1
    try:
        wcap = test_with(args.name, args.ext, "gray", args.img, args.crash)
        vidi.ffplay(wcap, loop=1, autoexit=True, fullscreen=False)
        print("%sTest [%d], record, success\n---------------%s"%(_s,_t,_n))
    except:
        _fail.append(1)
        print("%sTest [%d], record, failure\n---------------%s"%(_f,_t,_n))
    
    for i, f in enumerate(_fail):
        print("%sTest [%d], record, failure\n---------------%s"%(_f,f,_n))
    if not _fail:
        print("%sTests: success\n---------------%s"%(_s,_n))


if __name__ == "__main__":
    # test_cap("cap_avi", ".avi" 'rgb24', args.img)
    # test_cap("cap_mov_raw_gray", ".mov", 'raw', 'gray', args.img)

    # BUILD GRID SEARCH
    param_dic ={"ext":[".avi", "/mp4", ".mov"]}
    # grid_search()

    _num = int(args.num)
    if not _num:
        run_tests()

    elif _num == 1:
        # with:
        wcap = test_with(args.name, args.ext, args.pix_fmt, args.img, args.crash)
        vidi.ffplay(wcap, loop=1, autoexit=True, fullscreen=False)
    elif _num == 2:
        wcap = test_with(args.name, args.ext, "gray", args.img, args.crash)
        vidi.ffplay(wcap, loop=1, autoexit=True, fullscreen=False)
    else:
        cap = test_cap(args.name, args.ext, args.pix_fmt, args.img, args.crash)
        vidi.ffplay(cap, loop=1, autoexit=True, fullscreen=False)
