"""
pipe to numpy
((torch.from_numpy(myomy)).to(device='cuda').permute(2,0,1).to(dtype=torch.float)/255.).contiguous()
"""
import platform
import subprocess as sp
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import *
from .ff_fun import ffprobe
"""
TODO
make this work with dataloader
shift_batch ?
Detector: when does the frame shift, when does the shot change.

"""
_DEBUG = True

# class FTransform(object):
#     def __init__(self, name, mode, **params):
#         self.name = name
#         self.mode = mode
#         for p in **params:
#             self

#     def __call__(self):


class AVDataset(Dataset):
    def __init__(self, src, start=None, frames=None, step=1,
                 ftransform=None,
                 ptransform=None,
                 ntransform=None,
                 ttransform=None,
                 dtype=torch.float32, device="cpu", grad=False, debug=False):
        """
        Args
            src     (str)           ffmpeg readable video source
            start   (int [None])   start frame to start reading
                    (float [None]) start time (seconds)
            frames  (int [None])   number of frames to be read
                    (float [None]) time to be read
            step    (int [1]) step search between frames
            ftransform (list [None])   ffmpeg transforms  any filter in ffmpeg
            ptransform (object subclass [None])   PIL transforms
            ntransform (object subclass [None])   numpy transforms
            ttransform (object subclass [None])   torch transforms
            dtype (torch type ['float32])
            device  (str ['cpu']) cpu|cuda
            grad    (bool[False]) requires_grad
            debug   (bool[False])

        Examples:
            ftransform = ftransform=["edgedetect=high=0", "negate"]
            ftransform = ftransform=["edgedetect=mode=colormix:high=0"]
        """
        self.src = src
        self.stats = ffprobe(src)

        self.start = self._as_time(start)
        self.frames = self.stats["nb_frames"] if frames is None else self._as_frame(frames)
        self.step = step

        self.pix_fmt = "rgb24"
        self.dtype = dtype
        self.device = device
        self.grad = grad

        self.ftransform = ftransform
        self.ptransform = ptransform
        self.ntransform = ntransform
        self.ttransform = ttransform
        self.timer = None if not debug else Timer("FFread('%s')"%src)
        self.framecount = 0

        self._c = 1 if self.stats['pix_fmt'] == "gray" else 3
        self._h = self.stats["height"]
        self._w = self.stats["width"]
        self._bufsize = self._h * self._w * self._c

        self._pipe = None
        self._ffmpeg = 'ffmpeg' if platform.system() != 'Windows' else 'ffmpeg.exe'
        self._cmd = None

        self.debug = debug

    def __enter__(self):
        dprint('\n\t AVDataset.__enter__()')
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        dprint('\n\t AVDataset.__exit__()\n')
        self.close()

    def __len__(self):
        return self.frames

    def __getitem__(self, i=1):
        """ i can be used for ad hoc seek
            reads from buffer
            applies pil transforms
            applies numpy transforms
            applies torch transforms
        """
        if self._pipe is None or i is None or not i:
            return None

        # read ffmepeg frame
        try:
            for j in range(i):
                data = self._pipe.stdout.read(self._bufsize)
            dsubtic(self.timer, "read ffmpeg frame to buffer [%d]"%self.framecount)
        except:
            print("%sFailed to read buffer from command:\n\t%s%s%s"%(Col.YB, self._cmd, Col.RB, Col.AU))

        # PIL transforms
        if self.ptransform is not None:
            data = Image.frombuffer('RGB', (self._h, self._w), data, "raw", 'RGB', 0, 1)
            dsubtic(self.timer, "buffer to PIL [%d]"%self.framecount)
            data = np.array(self.ntransform(data)).reshape(1, self._h, self._w, self._c)
            dsubtic(self.timer, "PIL transforms [%d]"%self.framecount)

        else: # read as numpy
            data = np.frombuffer(data, dtype=np.uint8).reshape(1, self._h, self._w, self._c)
            dsubtic(self.timer, "buffer to numpy [%d]"%self.framecount)

        # numpy transforms
        if self.ntransform is not None:
            data = self.ntransform(data)
            dsubtic(self.timer, "numpy transforms [%d]"%self.framecount)

        # build tensor #1, 640, 3, 480]
        data = torch.from_numpy(data).to(device=self.device).permute(0, 3, 1, 2).to(dtype=self.dtype)
        if self.dtype in (torch.float16, torch.float32, torch.float64):
            data.div_(255.0)
        dsubtic(self.timer, "built tensor [%d]"%self.framecount)

        # torch transforms
        if self.ttransform is not None:
            data = self.ttransform(data)
            dsubtic(self.timer, "torch transforms [%d]"%self.framecount)

        if self.grad:
            data.requires_grad = True
        data.contiguous()

        dtic(self.timer, "to torch: ready [%d]"%self.framecount)
        self.framecount += 1
        return data

    def open(self):
        dtic(self.timer, "Open")
        if self._cmd is None:
            self._build_cmd()
        if self._pipe is not None:
            self.close()
        try:
            self._pipe = sp.Popen(self._cmd, stdout=sp.PIPE, bufsize=self._bufsize)
            dtic(self.timer, "pipe Popen")
        except:
            print("%sFailed to run ffmpeg command:\n\t%s%s%s"%(Col.YB, self._cmd, Col.RB, Col.AU))

    def close(self):
        if self._pipe is not None and not self._pipe.stdout.closed:
            self._pipe.stdout.flush()
            self._pipe.stdout.close()
            dtoc(self.timer, "pipe Popen")
        self._pipe = None

    def _build_cmd(self):
        self._cmd = [self._ffmpeg]

        # approximate search
        if self.start is not None:
            self._cmd += ["-ss", strftime(self.start-1)]

        #input
        self._cmd += ["-i", self.src]

        # precise search
        if self.start is not None:
            self._cmd += ["-ss", strftime(self.start)]

        #clip duration
        if self.frames is not None:
            self._cmd += ['-vframes', str(self.frames)]

        # ffmpeg filters
        _filters = ""
        if self.ftransform is not None:
            _filters = ",".join(self.ftransform)

        # step
        if self.step > 1:
            _step = "framestep=%d"%self.step
            if _filters:
                _filters = _step + "," + _filters
            else:
                _filters = _step

        if _filters:
            self._cmd += ["-vf", _filters]

        self._cmd += ["-f", "image2pipe", "-pix_fmt", self.pix_fmt, "-vcodec", "rawvideo"]
        self._cmd += ["-"]
        if self.debug:
            print(" ".join(self._cmd))

    def _as_frame(self, value):
        """int or float to int frame"""
        if value is None or isinstance(value, int):
            return value
        if isinstance(value, float): # interpret as time, return frame
            return time_to_frame(value, fps=self.stats["avg_frame_rate"])
        assert False, "%sexpected int (frames) or float (time), got %s%s"%(Col.RB, str(type(value)), Col.AU)

    def _as_time(self, value):
        """int or float to time"""
        if value is None or isinstance(value, float):
            return value
        if isinstance(value, int): # interpret as frame, return time
            return frame_to_time(value, fps=self.stats["avg_frame_rate"])
        assert False, "%sexpected int (frames) or float (time), got %s%s"%(Col.RB, str(type(value)), Col.AU)



# class FFdset(Dataset):

#     def __init__(self, src, batch_size=1, start=None, frames=None, pix_fmt="rgb24",
#                  dtype="float32", device="cpu", grad=False, shift_batch=False,
#                  debug=False):
#         """
#         Args
#             src     (str) valid video file
#             start   (number [None])
#                 if float: interpret as time
#                 if int: interpret as frame
#             frames  (number, [None])
#                 if float: interpret as time
#                 if int: interpret as frame
#             pix_fmt (str [rgb24])
#             debug   (bool [False])
#             dtype   (str [float32]) | float64 | float16 | uint8

#         Examples:

#             with vidi.FFread(src, batch_size=20, start=None, frames=1000,
#                              pix_fmt="rgb24, debug=True, dtype="float32", device="cpu", grad=False) as F:
#                 F.get_batch_loop()
#                 # F.get_batch() # slower than looping the reads Test multiprocessing


#         F.stats: {"index", "codec_name", "codec_type", "width", "height", "pix_fmt",
#                   "avg_frame_rate", "start_time", "duration", "nb_frames"}
#         """
#         self.src = src
#         self.stats = ffprobe(src)

#         self.start = self._as_time(start)
#         self.frames = self.stats["nb_frames"] if frames is None else self._as_frame(frames)
#         self.batch_size = batch_size
#         self._validate_batches()

#         self.pix_fmt = pix_fmt
#         self._c = 3
#         if self.stats['pix_fmt'] == "gray":
#             self._c = "1"
#             self.pix_fmt = "gray"
#         self._h = self.stats["height"]
#         self._w = self.stats["width"]
#         self._bufsize = self._h * self._w * self._c

#         self.dtype = torch.__dict__[dtype]
#         self.device = device
#         self.grad = grad

#         self.data = None
#         self.shift_batch = shift_batch
#         if shift_batch:
#             self.data = self._init_data()

#         self.debug = debug
#         self.timer = None if not debug else Timer("FFread(src, batch_size=%d)"%(self.batch_size))
#         self._cmd = None
#         self._pipe = None
#         self._ffmpeg = 'ffmpeg' if platform.system() != 'Windows' else 'ffmpeg.exe'
#         self.framecount = 0

#         self._div = 255.0 if 'float' in self.dtype else 1

#     def __enter__(self):
#         print('\n\t .__enter__()')
#         self.open()
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         if self.debug:
#             print('\n\t .__exit__()\n')
#         self.close()

#     def __len__(self):
#         return self.frames

#     def __getitem__(self, i):
#         if self._pipe is None:
#             return None

#         try:
#             data = np.frombuffer(self._pipe.stdout.read(self._bufsize), dtype=np.uint8).reshape(1, self._h, self._w, self._c)
#         except:
#             print("%sFailed to get buffer from command:\n\t%s%s%s"%(Col.YB, self._cmd, Col.RB, Col.AU))

#         if self.timer is not None:
#             self.timer.subtic("subtic, from buffer [%d]"%self.framecount)

#         data = (torch.from_numpy(data).to(device=self.device).permute(0, 3, 1, 2).to(dtype=self.dtype)/self._div).contiguous()
#         if self.timer is not None:
#             self.timer.tic("to torch [%d]"%self.framecount)

#         self.framecount += 1

#         # todo make this work with dataloader ?
#         if self.shift_batch:
#             self.data[1:] = self.data[:-1]
#             self.data[-1] = data[0]

#         return data

#     def open(self):
#         if self.timer is not None:
#             self.timer.tic("Open")
#         if self._cmd is None:
#             self._build_cmd()
#         if self._pipe is not None:
#             self.close()
#         self._pipe = sp.Popen(self._cmd, stdout=sp.PIPE, bufsize=self._bufsize)
#         if self.timer is not None:
#             self.timer.tic("pipe Popen")

#     def close(self):
#         if self._pipe is not None and not self._pipe.stdout.closed:
#             self._pipe.stdout.flush()
#             self._pipe.stdout.close()
#             if self.timer is not None:
#                 self.timer.toc("pipe closed")
#         self._pipe = None

#     def _init_data(self):
#         return torch.zeros([self.batch_size, self._c, self._h, self._w],
#                            dtype=self.dtype, device=self.device, requires_grad=self.grad)

#     def _build_cmd(self):

#         self._cmd = [self._ffmpeg]

#         # approximate search
#         if self.start is not None:
#             self._cmd += ["-ss", strftime(self.start-1)]

#         self._cmd += ["-i", self.src]
#         # precise search
#         if self.start is not None:
#             self._cmd += ["-ss", strftime(self.start)]

#         #clip duration
#         if self.frames is not None:
#             self._cmd += ['-vframes', str(self.frames)]

#         self._cmd += ["-f", "image2pipe", "-pix_fmt", self.pix_fmt, "-vcodec", "rawvideo"]
#         self._cmd += ["-"]

#     def _as_frame(self, value):
#         """int or float to int frame"""
#         if value is None or isinstance(value, int):
#             return value
#         if isinstance(value, float): # interpret as time, return frame
#             return time_to_frame(value, fps=self.stats["avg_frame_rate"])
#         assert False, "%sexpected int (frames) or float (time), got %s%s"%(Col.RB, str(type(value)), Col.AU)

#     def _as_time(self, value):
#         """int or float to time"""
#         if value is None or isinstance(value, float):
#             return value
#         if isinstance(value, int): # interpret as frame, return time
#             return frame_to_time(value, fps=self.stats["avg_frame_rate"])
#         assert False, "%sexpected int (frames) or float (time), got %s%s"%(Col.RB, str(type(value)), Col.AU)

#     def _validate_batches(self):
#         assert self.frames >= self.batch_size, "%srequesting batch size (%d) larger than number of frames in video: (%d)%s"%(Col.RB, self.batch_size, self.frames, Col.AU)
#         _frames = self.batch_size*(self.frames//self.batch_size)
#         if _frames < self.frames:
#             print("%sskipping last %d frames from batch%s"%(Col.YB, (self.frames - _frames), Col.AU))
#             self.frames = _frames

