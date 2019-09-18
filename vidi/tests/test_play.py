import os
from unittest import TestCase

import vidi
from vidi.utils import Col

class TestPlay(TestCase):

    def test_playavi(self):
        print(Col.GB+"test_playavi(): vidi.ffplay() an avi"+Col.YB)
        print(Col.GB, "Test folder", os.getcwd(), Col.AU)
        src = "test_data/vidi_test_with_gray.avi"
        vidi.ffplay(src, folder=None, start=0, fps=None, loop=1, autoexit=True, fullscreen=False)

    def test_playseq(self):
        print(Col.GB+"test_playseq(): vidi.ffplay() on sequence: \n\tsrc = 'out%08d.png' \n\tfolder='test_data/seq'"+Col.AU)
        folder = "test_data/seq"
        src = "out%08d.png"
        vidi.ffplay(src, folder=folder, start=0, fps=15, loop=1, autoexit=True, fullscreen=False)

    def test_playimgs(self):
        print(Col.GB+"test_playimgs(): vidi.ffplay() on imgs \n\tsrc = 'test_data/imgs/*.jpg'"+Col.AU)

        folder = None
        src = "test_data/imgs/*.jpg"
        vidi.ffplay(src, folder=folder, start=0, fps=30, loop=1, autoexit=True, fullscreen=False)
