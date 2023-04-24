"""@xvdp
util functions for vidi
    Col                     print colors
    frame_to_time(frame, fps)       -> seconds
    frame_to_strftime(frame, fps)   -> HH:MM:SS.mmm
    stftime(seconds)                -> HH:MM:SS.mmm
    strftime_to_time(HH:MM:SS.mmm)  -> seconds
    anytime_to_frame_time((HH:MM:SS.mmm, frame, secinds), fps) -> (frame, seconds)
"""
from typing import Optional, Union
import subprocess as sp
import time
import psutil



class Col:
    """colorization shortucts"""
    AU = '\033[0m'
    BB = '\033[94m\033[1m'
    GB = '\033[92m\033[1m'
    YB = '\033[93m\033[1m'
    RB = '\033[91m\033[1m'
    B = '\033[1m'

###
#
# frame, time, strftime conversions
#

def frame_to_time(frame: int, fps: float) -> float:
    """frame -> seconds (int to float)
    Args
        frame   int
        fps     float
    """
    outtime = frame/fps
    return outtime


def frame_to_strftime(frame: int, fps: float) -> str:
    """frame -> HH:MM:SS.mmm, (int to str)
    Args
        frame   int     frame number
        fps     float   frame rate
    """
    return strftime(frame_to_time(frame, fps))


def time_to_frame(intime: float, fps: float) -> int:
    """ seconds -> frame, (float to int)
    Args
        intime  float seconds
        fps     float
    """
    return int(round(intime * fps))


def strftime(intime: float) -> str:
    """seconds -> HH:MM:SS.mmm"""
    _t = int(intime)
    return f"{(_t//3600)%24:02d}:{(_t//60)%60:02d}:{_t%60:02d}.{(int((intime - _t)*1000)):03d}"


def strftime_to_time(instftime: str) -> float:
    """ HH:MM:SS.mmm -> seconds
    """
    return round(sum(x * float(t) for x, t in zip([3600, 60, 1], instftime.split(":"))), 3)


def anytime_to_frame_time(intime: Union[int, float, str],
                          fps: float) -> tuple[int, float]:
    """ returns (frame int, time float)
    Args
        intime     (str 'HH:MM:SS.mmm', float, int)
        fps         (float)
    """
    if isinstance(intime, str):
        intime = strftime_to_time(intime)

    if isinstance(intime, float):
        out_time = intime
        out_frame = time_to_frame(intime, fps)
    else: # int
        out_frame = intime
        out_time = frame_to_time(intime, fps)
    return out_frame, out_time


###
#
# Timer TODO revise. unused
#
class Timer:
    """ simple timing class
    """
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
                print(f" Time:({name} {color}\t{indent}{self.times[name]*1000,:.3} ms{Col.AU})")

    def subtic(self, name: str, color: str = Col.GB, indent: str = "  ") -> None:
        """
        computes time diff with .last
        """
        _now = time.time()
        if name is not None:
            while name in self.times:
                name = name+"0"
            self.times["subtic:"+name] = _now - self.last
            if self.live_tics and name is not None:
                print(f" Time:({name} {color}\t{indent}{self.times[name]*1000:.3}ms{Col.AU})")

    def toc(self, name: Optional[str] = None):
        self.tic(name)

        name = self.name if self.name is not None else ""
        print(f"{Col.GB}Time: {name}{Col.AU}")

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
            print(f" {_name}{col}\t{self.times[_t]*1000:.3} ms{Col.AU}")

        # print total
        self.times["total"] = self.now - self.start
        if self.times["total"] >= 60:
            _total = strftime(self.times["total"])
        else:
            _total = f"{(self.times['total']*1000):.3} ms"

        _name = "Total\t"+" "*max(0, _len-len("Total"))
        print(f"{Col.YB}{_name}\t{_total}{ Col.AU}")


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


###
# TODO revise, newer functions available
#

def get_smi(query):
    """reutrn nvidia-smi query"""
    _cmd = ['nvidia-smi', f'--query-gpu=memory.{query}', '--format=csv,nounits,noheader']
    return int(sp.check_output(_cmd, encoding='utf-8').split('\n')[0])

class GPUse:
    """thin wrap to nvidia-smi"""
    def __init__(self, units="MB"):
        self.total = get_smi("total")
        self.used = get_smi("used")
        self.available = self.total - self.used
        self.percent = round(100*self.used/self.total, 1)
        self.units = units if units[0].upper() in ('G', 'M') else 'MB'
        self._fix_units()

    def _fix_units(self):
        if self.units[0].upper() == "G":
            self.units = "GB"
            self.total //= 2**10
            self.used //= 2**10
            self.available //= 2**10

    def __repr__(self):
        return f"GPU: ({self.__dict__})"

class CPUse:
    """thin wrap to psutil.virtual_memory to matching nvidia-smi syntax"""
    def __init__(self, units="MB"):
        cpu = psutil.virtual_memory()
        self.total = cpu.total
        self.used = cpu.used
        self.available= cpu.available
        self.percent = cpu.percent
        self.units = units if units[0].upper() in ('G', 'M') else 'MB'
        self._fix_units()

    def _fix_units(self):
        _scale = 20
        if self.units[0].upper() == "G":
            self.units = "GB"
            _scale = 30
        else:
            self.units = "MB"
        self.total //= 2**_scale
        self.used //= 2**_scale
        self.available //= 2**_scale

    def __repr__(self):
        return f"CPU: ({self.__dict__})"
