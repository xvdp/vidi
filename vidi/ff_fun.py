"""
ffmpeg, ffplay, ffprobe wrapper using pipes
functions file

"""
import os
import os.path as osp
import glob
import subprocess as sp
from platform import system

from .utils import *
from .ff_main import FF

__all__ = ["ffplay", "ffstitch", "ffprobe", "ffread"]

def ffprobe(src, entries=None, verbose=False):
    """ Wrapper for ffprobe, returning size, frame rate, number of frames
    Args
        src, video
        entries, subsection of _entries

    more info https://trac.ffmpeg.org/wiki/FFprobeTips
    """
    assert osp.isfile(src), "%sinexistent file <%s>%s"%(Col.RB, src, Col.AU)

    _entries = ["index", "codec_name", "codec_type", "width", "height", "pix_fmt",
                "avg_frame_rate", "start_time", "duration", "nb_frames"]
    _video_entries = ["avg_frame_rate", "start_time", "duration", "nb_frames"]

    if entries is not None:
        if isinstance(entries, str):
            entries = [entries]
        _entries = _ffprobe_sort_entries(entries, _entries, verbose)

    _v = "info" if verbose else "error"
    _fcmd = ["ffprobe", "-v", _v]
    _fcmd += ["-show_entries", "stream="+",".join(_entries)]
    _fcmd += ["-of", "default=noprint_wrappers=1:nokey=1"]
    _fcmd += [src]
    _fcmd = " ".join(_fcmd)

    if verbose:
        print("%s%s%s"%(Col.GB, _fcmd, Col.AU))

    if verbose:
        T = Timer(_fcmd)

    try:
        _pipe = os.popen(_fcmd)
        if verbose:
            T.tic("open ffprobe pipe")

        stats = {}

        try:
            for i, _e in enumerate(_entries):
                stats[_e] = _ffprobe_parse_stat(_pipe.readline())
                if verbose:
                    T.tic("read <%s>"%_e)

        except:
            if not verbose:
                print("%sfailed at read line %s on %s%s"%(Col.RB, _e, _fcmd, Col.AU))
            stats['file'] = src
    except:
        print("%sCannot run ffprobe on <%s>%s"%src)

    if verbose:
        T.toc()
    _pipe.close()
    #_pipe.terminate()
    return stats

def ffplay(src, start=0, fps=None, loop=0, autoexit=True, fullscreen=False, noborder=True):
    """
    Args:
        src (str)
            *.png
            image%08.png
            movie.mov

        folder (str), if None, plays current folder
        start: (number) if file sequences: start frame, if movie, start time
        fps:
        loop    (int[0]) : number of loops, 0: forever

    Examples:
        >>> vidi.ffplay("metro%08d.png", start=9000) # patterned files local folder
        >>> vidi.ffplay("*.png")   # pngs local folder
        >>> vidi.ffplay(""~/mypath/metro%08d.png", fps=100, start=5)
        >>> vidi.ffplay(""~/mypath") # files in folder - if only one extension type found
        >>> vidi.ffplay("~/metro_color.mov")
    """
    src = _format_src(src)

    fcmd = ["ffplay"]

    if '%' in src or '*' in src:
        if fps is None:
            fps = 29.97
        fcmd += ["-framerate", str(fps)]
    elif fps is not None:
        print("%sfps not supported for <%s>%s"%(Col.YB, src, Col.AU))

    # file lists and folders
    fcmd += _ff_format_input_list(src, start)
    fcmd += ["-loop", str(loop)]
    if autoexit:
        fcmd += ["-autoexit"]
    if noborder:
        fcmd += ["-noborder"]
    if fullscreen:
        fcmd += ["-fs"]
    # if alwaysontop: # not on ubuntu 18
    #     fcmd += ["-alwaysontop"]
    fcmd += ["-i", src]

    _ffplaymsg(fcmd, msg="Running ffplay on folder")

    sp.call(fcmd)

def ffstitch(src, dst, fps=29.97, start=0, size=None, num=None, audio=None):
    """ stitches folder of files to video
    TODO wrap ff_main instead
    Args
       required:
        src         (str)   "*.<ext>" | "<name>%0<pad>d.<ext>" | foldername
        dst         (str)   "<name>.<ext>"
       optional:
        fps         (float [29.97]) frames per second
        start       (int [0]) start frame
        size        ((int,int) [None]), resize output to
        num         (int [None]) number of frames
        audio       (str [None]) path to audio file


    Examples
        vidi.ffstitch("*.png", "metro_color_tb4.mov", num=100)
        vidi.ffstitch("metro%08d.png", "metro_color_tb4.mov", num=100, size=50)
        vidi.ffstitch("metro%08d.png", "metro_color_tb4.mov", start=6000, num=100, size=(600,200))

        vidi.ffstitch("metro%08d.png", "metro_color_tb4.mov", audio="/home/z/data/Music/Kronos/KNOX_-_SATELLITES_-_FULL_SCORE_AND_PARTS_1.aac", start=6000, num=2000)

        #ffmpeg -r 29.97 -i "metro%08d.png" -start_number 6468 -vframes 200 -vcodec libx264 -pix_fmt yuv420p /home/z/metro_color.mov
    """
    src = _format_src(src)

    print("-------------Running vidi.ffstitch()-------------")

    _ff = 'ffmpeg' if system() != "Windows" else 'ffmpeg.exe'

    # frame rate
    _fcmd = [_ff, '-r', str(fps)]

    # file list format
    _fcmd += _ff_format_input_list(src, start)
    _fcmd += ['-i', src]

    # audio
    if audio is not None:
        _fcmd += ['-i', audio]

    size = _ff_format_size(size)
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
    print(Col.GB + " ".join(_fcmd) + Col.AU)
    print(" -------------------------------------")
    sp.call(_fcmd)

    return dst

def ffread(src):
    """ converts to numpy using FF class
    Args
        src, valid video file
    TODO: RAM checking - this is brute force conversion and if video is too large will explode ram
    TODO: could include cropping
    """
    assert osp.isfile(src), f"{src} not found"
    # ext = osp.splitext(src)[1].lower()
    # if ext not in ('.mkv', '.avi', '.mp4', '.mov', '.flv'):
    #     raise NotImplementedError(f"video format not supported {}")
    return FF(src).to_numpy()



def _ffprobe_parse_stat(stat):
    """if stat.isnumeric(): only handles ints"""
    stat = stat.replace('\n', '')
    try:
        stat = eval(stat)
    except:
        pass
    return stat

def _ffprobe_sort_entries(requested, dictionary, verbose=False):
    out = [_e for _e in dictionary if _e in requested]
    _fail = [_e for _e in requested if _e not in out]
    if verbose:
        print("%sRequested entries found: %s%s"%(Col.GB, str(out), Col.AU))
    if _fail:
        print("%sRequested entries not found: %s%s"%(Col.RB, str(_fail), Col.AU))
    return out


def _ffplaymsg(fcmd="", msg=""):
    print(Col.B, " ".join(fcmd))
    print("-------Interaction--------")
    print(" 'q', ESC        Quit")
    print(" 'f', LMDC       Full Screen")
    print(" 'p', SPACE      Pause")
    print(" '9'/'0'         Change Volume")
    print(" 's'             Step one frame")
    print("  RT,DN / LT,UP  10sec jump")
    print("  RMC            jump to percentage of film")
    print("--------------------------", Col.AU)

def _format_src(src):
    src = osp.abspath(osp.expanduser(src))
    if osp.isdir(src):
        files = sorted([f.name for f in os.scandir(src)
                        if osp.splitext(f.name)[1].lower() in (".jpg", ".jpeg", ".png")])
        exts = list(set([osp.splitext(f)[1] for f in files]))
        assert len(exts) == 1, f"one and only one extension supported found {exts}"
        src = osp.join(src, f"*{exts[0]}")
    return src

def _ff_check_sequence(src, start=0):
    """ 
    Validate first frame of a sequence, and sequence exist
    Return None if invalid, <int> if valid
    """
    if osp.isfile(src%start):
        return start

    _pattern = src.split("%")
    _front = _pattern[0]
    _back = "."+".".join(_pattern[1].split('.')[1:])
    _pattern = _front+"*"+_back

    _files = sorted(glob.glob(_pattern))#[0].split(_front)[1].split(_back)[0]

    if not _files:
        print("%sno files found%s"%(Col.RB, Col.AU))
        return None

    print("%s%d files %s%s"%(Col.GB, len(_files), src, Col.AU))
    first_frame = int(_files[0].split(_front)[1].split(_back)[0])
    return first_frame



def _ff_format_input_list(src, start=0):
    """Format name to ffmpeg str
    Args
        src     (str)     input
        start   (int [0]) start frame
    """
    fcmd = []
    # patterned file sequence, eg. metro%0d.png
    if '%' in src:
        _start = _ff_check_sequence(src, start)
        assert _start is not None, "%sno files with pattern <%s> found%s"%(Col.RB, src, Col.AU)
        if _start != start:
            print("%sstart frame <%d> not found, using <%d> instead%s"%(Col.YB, start, _start, Col.AU))
            start = _start
        fcmd += ["-start_number", str(start)]

    # files of pattern in folder e.g. *.png or metro*.png
    elif '*' in src:
        if start is not None:
            print("%signoring 'start' argument, only valid with templated arguments%s"%(Col.YB, Col.AU))
        fcmd += ["-pattern_type", "glob"]
    elif start and start is not None:
        if isinstance(start, int): # interpret as frame, return time
            start = frame_to_time(start, fps=29.97) #self.stats["avg_frame_rate"]
        fcmd += ["-ss", strftime(start-1)]

    return fcmd

def _ff_format_size(size=None):
    """Format size to ffmpeg str: "%dx%d"%size
    Args
        size (int, (int, int))

    """
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
