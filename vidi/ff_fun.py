import os
import os.path as osp
import glob
import subprocess as sp
from platform import system
from .io_main import IO
import json


class _C_:
    AU = '\033[0m'
    BB = '\033[94m\033[1m'
    GB = '\033[92m\033[1m'
    YB = '\033[93m\033[1m'
    RB = '\033[91m\033[1m'
    B = '\033[1m'


def ffplaymsg(fcmd="", msg=""):
    print(_C_.B, " ".join(fcmd))
    print("-------Interaction--------")
    print(" 'q', ESC        Quit")
    print(" 'f', LMDC       Full Screen")
    print(" 'p', SPACE      Pause")
    print(" '9'/'0'         Change Volume")
    print(" 's'             Step one frame")
    print("  RT,DN / LT,UP  10sec jump")
    print("  RMC            jump to percentage of film")
    print("--------------------------", _C_.AU)

def check_seq_file(src, start=0, folder=None):
    if folder is not None:
        src = osp.join(folder, src)

    if osp.isfile(src%start):
        return start

    _pattern = src.split("%")
    _front = _pattern[0]
    _back = "."+".".join(_pattern[1].split('.')[1:])
    _pattern = _front+"*"+_back


    _files = sorted(glob.glob(_pattern))#[0].split(_front)[1].split(_back)[0]

    if not _files:
        print("%sno files found%s"%(_C_.RB, _C_.AU))
        return None

    print("%s%d files %s%s"%(_C_.GB, len(_files), src, _C_.AU))
    lowest = int(_files[0].split(_front)[1].split(_back)[0])
    return lowest


def resolve_name(src, start=0, folder=None):

    fcmd = []
    # patterned file sequence, eg. metro%0d.png
    if '%' in src:
        _start = check_seq_file(src, start, folder)
        assert _start is not None, "%sno files with pattern <%s> found%s"%(_C_.RB, src, _C_.AU)
        if _start != start:
            print("%sstart frame <%d> not found, using <%d> instead%s"%(_C_.YB, start, _start, _C_.AU))
            start = _start
        fcmd += ["-start_number", str(start)]

    # files of pattern in folder e.g. *.png or metro*.png
    elif '*' in src:
        if start is not None:
            print("%signoring 'start' argument, only valid with templated arguments%s"%(_C_.YB, _C_.AU))
        fcmd += ["-pattern_type", "glob"]

    return fcmd

def get_size(size=None):
    if size is not None:
        if isinstance(size, str):
            if 'x' not in size:
                assert size.isnumeric(size), "size can be tuple of ints, string of ints sepparated by 'x' or single int"
                size = int(size)
        if isinstance(size, int):
            size = (size, size)
        if isinstance(size, (tuple, list)):
            size = "%dx%d"%size
    return size

def ffplay(src, folder=None, start=0, fps=None):
    """
    Args:
        src str
            *.png
            image%08.png
            movie.mov

        folder str, if None, plays current folder
        start: if file sequences: start frame, if movie, start time

    Examples:
    vidi.ffplay("metro%08d.png", start=9000)
    vidi.ffplay("*.png")
    vidi.ffplay("metro%08d.png", folder="~/work/gans/pix2pix/results/color_kaiming", fps=100, start=5)
    vidi.ffplay("metro_color.mov", folder="~") #, fps=100, start=5 not implemneted for mov
    """
    _cwd = os.getcwd()
    if folder is not None:
        folder = osp.abspath(osp.expanduser(folder))
        assert osp.isdir(folder), "%sfolder <%s> does not exist%s"%(_C_.RB, folder, _C_.AU)
        os.chdir(folder)

    fcmd = ["ffplay"]

    if '%' in src or '*' in src:
        if fps is None:
            fps = 29.97
        fcmd += ["-framerate", str(fps)]
    elif fps is not None:
        print("%sfps not supported for <%s>%s"%(_C_.YB, src, _C_.AU))

    # file lists and folders
    fcmd += resolve_name(src, start, folder)

    fcmd += ["-i", src]

    ffplaymsg(fcmd, msg="Running ffplay on folder")

    sp.call(fcmd)

    if folder is not None:
        os.chdir(_cwd)


def ffstitch(src, dst, folder=None, fps=29.97, start=0, size=None, num=None, audio=None): # dst, src, audio, fps, size, start_img, max_img)
    """
    Args
       required:
        src
        dst
       optional:
        folder
        fps
        start
        size
        num
        audio


    Examples
        vidi.ffstitch("*.png", "metro_color_tb4.mov", num=100)
        vidi.ffstitch("metro%08d.png", "metro_color_tb4.mov", num=100, size=50)
        vidi.ffstitch("metro%08d.png", "metro_color_tb4.mov", start=6000, num=100, size=(600,200))

        vidi.ffstitch("metro%08d.png", "metro_color_tb4.mov", audio="/home/z/data/Music/Kronos/KNOX_-_SATELLITES_-_FULL_SCORE_AND_PARTS_1.aac", start=6000, num=2000)

        #ffmpeg -r 29.97 -i "metro%08d.png" -start_number 6468 -vframes 200 -vcodec libx264 -pix_fmt yuv420p /home/z/metro_color.mov
    """
    _cwd = os.getcwd()

    if folder is not None:
        folder = osp.abspath(osp.expanduser(folder))
        assert osp.isdir(folder), "%sfolder <%s> does not exist%s"%(_C_.RB, folder, _C_.AU)
        os.chdir(folder)

    print("-------------Running vidi.ffstitch()-------------")

    _ff = 'ffmpeg' if system() != "Windows" else 'ffmpeg.exe'
    _fcmd = [_ff, '-r', str(fps)]

    _fcmd += resolve_name(src, start)#, folder)

    _fcmd += ['-i', src]

    if audio is not None:
        _fcmd += ['-i', audio]

    size = get_size(size)
    if size is not None:
        _fcmd += ['-s', size]

    # codecs
    _fcmd += ['-vcodec', 'libx264']
    if audio is not None:
        _fcmd += ['-acodec', 'copy']

    # format
    _fcmd += ['-pix_fmt', 'yuv420p']

    # number of frames # has to be just before outpyut
    if num is not None:
        _fcmd += ['-vframes', str(num)]

    _fcmd += [dst]
    print(_C_.GB + " ".join(_fcmd) + _C_.AU)
    print(" -------------------------------------")
    sp.call(_fcmd)

    if folder is not None:
        os.chdir(_cwd)
        dst = osp.join(folder, dst)
    return dst
