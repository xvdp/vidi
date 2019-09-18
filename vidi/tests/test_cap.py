import os
import os.path as osp
from unittest import TestCase

import numpy as np
from PIL import Image
import vidi
from vidi.utils import Col

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
        if crash and i == 2:
            return
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



IMG = get_img("test_data/bundaberg_bully.jpg")
SIZE = IMG.shape[:2]
NAME = "test_cap.avi"
CRASH = True

class TestCap(TestCase):

    def test_Capture_RGB(self):

        print(Col.GB+"test_capture() RGB"+Col.YB)
        print(Col.GB, "Test folder", os.getcwd(), Col.AU)

        pix_fmt = "rgb24"
        NAME = "test_data/test_cap.avi"
        clean_test(NAME)

        with vidi.FFcap(NAME, pix_fmt=pix_fmt, size=SIZE, overwrite=True, debug=True) as F:
            record(F, IMG, CRASH, SIZE, pix_fmt)

        return NAME

    def test_Capture_GRAY(self):
        print(Col.GB+"test_capture() GRAY"+Col.YB)

        pix_fmt = "gray"
        NAME = "test_data/test_cap_gray.avi"
        clean_test(NAME)


        with vidi.FFcap(NAME, pix_fmt=pix_fmt, size=SIZE, overwrite=True, debug=True) as F:
            record(F, IMG, CRASH, SIZE, pix_fmt)

        return NAME

