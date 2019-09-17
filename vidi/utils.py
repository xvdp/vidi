import time
import numpy as np
import torch
class Col:
    AU = '\033[0m'
    BB = '\033[94m\033[1m'
    GB = '\033[92m\033[1m'
    YB = '\033[93m\033[1m'
    RB = '\033[91m\033[1m'
    B = '\033[1m'

class Timer:
    def __init__(self, name=None, live_tics=False):
        self.name = name
        self.live_tics = live_tics
        self.start = time.time()
        self.last = time.time()
        self.now = time.time()
        self.times = {}

    def tic(self, name=None, color=Col.BB, indent=" "):
        """
        updates .now and, computes time diff with .last, updates .last
        """
        self.now = time.time()
        if name is not None:
            while name in self.times:
                name = name+"0"
            self.times[name] = self.now - self.last
            self.last = time.time()

            if self.live_tics and name is not None:
                print(" Timer:(%s %s\t%s%.3f ms%s)"%(name, color, indent, self.times[name]*1000, Col.AU))

    def subtic(self, name, color=Col.GB, indent="  "):
        """
        computes time diff with .last
        """
        _now = time.time()
        if name is not None:
            while name in self.times:
                name = name+"0"
            name = name
            self.times["subtic:"+name] = _now - self.last

            if self.live_tics and name is not None:
                print(" Timer:(%s %s\t%s%.3f ms%s)"%(name, color, indent, self.times[name]*1000, Col.AU))


    def toc(self, name=None):
        self.tic(name)

        name = self.name if self.name is not None else ""
        print("%sTimer: %s%s"%(Col.GB, name, Col.AU))

        # print tics
        _len = len(sorted(self.times, key=len)[-1])
        _i = list(self.times.values()).index(max(self.times.values()))

        for i, _t in enumerate(self.times):
            indent = "\t"
            col = Col.BB if i != _i else Col.RB
            _name = _t
            if "subtic" in _t:
                _name = _name.split("subtic")[1]
                col = Col.GB
                indent = ""

            _name = _name + (_len - len(_name))*" "+ indent
            print(" %s%s\t%.3f ms%s"%(_name, col, self.times[_t]*1000, Col.AU))

        # print total
        self.times["total"] = self.now - self.start
        if self.times["total"] >= 60:
            _total = strftime(self.times["total"])
        else:
            _total = "%.3f ms"%(self.times["total"]*1000)
        _name = "Total\t"+" "*max(0, _len-len("Total"))
        print("%s%s\t%s%s"%(Col.YB, _name, _total, Col.AU))

def dprint(*msg, debug=False, **kwmsg):
    """debug print wrapper"""
    if debug:
        print(*msg, **kwmsg)

def dtic(timer, msg):
    """conditional wrapper to Timer().tic"""
    if timer is not None:
        timer.tic(msg)
def dsubtic(timer, msg):
    """conditional wrapper to Timer().subtic"""
    if timer is not None:
        timer.subtic(msg)
def dtoc(timer, msg):
    """conditional wrapper to Timer().toc"""
    if timer is not None:
        timer.toc(msg)

def frame_to_time(frame, fps):
    """convert frame number to time"""
    outtime = frame/fps
    return outtime

def time_to_frame(intime, fps):
    frame = int(intime * fps)
    return frame

def strftime(intime):
    return '%02d:%02d:%02d.%03d'%((intime//3600)%24, (intime//60)%60, intime%60, (int((intime - int(intime))*1000)))

def tofloat32(uint8_value):
    return uint8_value/np.array(255, dtype=np.float32)

def validate_dtype(dtype, as_torch=False):
    if dtype in ("float", "float32"):
        dtype = "float32"
    elif dtype in ("double", "float64"):
        dtype = "float64"
    elif dtype in ("half", "float16"):
        dtype = "float16"
    elif dtype in ("uint", "uint8"):
        dtype = "uint8"
    else:
        assert "dtype <%s> not recognized"%dtype
    if as_torch:
        dtype = torch.__dict__[dtype]
    return dtype
