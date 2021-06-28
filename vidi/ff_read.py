"""
# TODO: cleanup and deprecate
pipe to numpy
"""
import platform
import subprocess as sp
import numpy as np
import torch

from .utils import *
from .ff_fun import ffprobe

class FFread:

    def __init__(self, src, batch_size=1, out_type="numpy", start=None, frames=None,
                 pix_fmt="rgb24", dtype="float32", device="cpu", grad=False, debug=False):
        """
        Args
            src     (str) valid video file
            start   (number [None])
                if float: interpret as time
                if int: interpret as frame
            frames  (number, [None])
                if float: interpret as time
                if int: interpret as frame
            pix_fmt (str [rgb24])
            debug   (bool [False])
            dtype   (str [float32]) | float64 | float16 | uint8

        Examples:

            with vidi.FFread(src, batch_size=20, out_type="torch", start=None, frames=1000,
                             pix_fmt="rgb24, debug=True, dtype="float32", device="cpu", grad=False) as F:
                F.get_batch_loop()
                # F.get_batch() # slower than looping the reads Test multiprocessing


        F.stats: {"index", "codec_name", "codec_type", "width", "height", "pix_fmt",
                  "avg_frame_rate", "start_time", "duration", "nb_frames"}
        """
        self.src = src
        self.stats = ffprobe(src)

        self.start = self._as_time(start)
        self.frames = self.stats["nb_frames"] if frames is None else self._as_frame(frames)
        self.batch_size = batch_size
        self._validate_batches()

        # TODO, pix_fmt handle
        self.pix_fmt = pix_fmt # yuv420p
        self._c = 3
        if self.stats['pix_fmt'] == "gray":
            self._c = "1"
            self.pix_fmt = "gray"
        self._h = self.stats["height"]
        self._w = self.stats["width"]
        self._bufsize = self._h * self._w * self._c

        self.out_type = out_type
        self.dtype = validate_dtype(dtype)
        self.device = device
        self.grad = grad

        self.data = self._init_data()

        self.debug = debug
        self.timer = None if not debug else Timer("FFread(src, batch_size=%d, out_type=%s)"%(self.batch_size, self.out_type))
        self._cmd = None
        self._pipe = None
        self._ffmpeg = 'ffmpeg' if platform.system() != 'Windows' else 'ffmpeg.exe'
        self.framecount = 0
        self._div = 1
        if self.dtype in "float":
            self._div = 255.0
        self._255 = np.array(255, dtype=self.dtype)

    def __enter__(self):
        print('\n\t .__enter__()')
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.debug:
            print('\n\t .__exit__()\n')
        self.close()
        del self.data

    def open(self):
        if self.timer is not None:
            self.timer.tic("Open")
        if self._cmd is None:
            self._build_cmd()
        if self._pipe is not None:
            self.close()
        self._pipe = sp.Popen(self._cmd, stdout=sp.PIPE, bufsize=self._bufsize)
        if self.timer is not None:
            self.timer.tic("pipe Popen")

    def close(self):
        if self._pipe is not None and not self._pipe.stdout.closed:
            self._pipe.stdout.flush()
            self._pipe.stdout.close()
            if self.timer is not None:
                self.timer.toc("pipe closed")
        self._pipe = None

    def get_batch(self):
        """ Fills batch data by reading from ffmpeg buffer, one batch at a time
        """
        self.data *= 0
        self._update_data(self._pipe.stdout.read(self._bufsize * self.batch_size))

        self.framecount += self.batch_size

        if self.timer is not None:
            self.timer.tic("batch retrieved, framecount [%d]"%self.framecount)

        if self.framecount >= self.frames:
            return self.close()

    def _update_data(self, data):

        _framecount = self.framecount+self.batch_size

        _data = np.frombuffer(data, dtype=np.uint8)
        if self.timer is not None:
            self.timer.subtic("update from buffer [%d]"%_framecount)

        _data = _data.reshape(self.batch_size, self._h, self._w, self._c)
        if self.timer is not None:
            self.timer.subtic("reshaped to (%d, %d, %d, %d) [%d]"%(self.batch_size, self._h, self._w, self._c, _framecount))


        if self.out_type == "numpy":
            if "float" in self.dtype:
                _data = _data/self._255
            if self.timer is not None:
                self.timer.subtic("to dtype %s, [%d]"%(self.dtype, _framecount))
            self.data[:] = _data
            if self.timer is not None:
                self.timer.subtic("fill np array [%d]"%(_framecount))
        else:
            self.data[:] = ((torch.from_numpy(_data).to(device=self.device).permute(0, 3, 1, 2)).to(dtype=torch.__dict__[self.dtype])/self._div).contiguous()
            if self.timer is not None:
                self.timer.subtic("fill tensor and permuted [%d]"%(_framecount))

    def get_batch_loop(self):
        """ Fills batch data by reading from ffmpeg buffer, one image at a time
        """
        self.data *= 0

        for i in range(self.batch_size):
            self._update_data_loop(i, self._pipe.stdout.read(self._bufsize))

        self.framecount += self.batch_size

        if self.timer is not None:
            self.timer.tic("batch retrieved, framecount [%d]"%self.framecount)

        if self.framecount >= self.frames:
            return self.close()

    def _update_data_loop(self, idx, data):

        _data = np.frombuffer(data, dtype=np.uint8)
        if self.timer is not None:
            self.timer.subtic("[%d]update from buffer "%(self.framecount+idx))

        _data = _data.reshape(self._h, self._w, self._c)
        if self.timer is not None:
            self.timer.subtic("[%d] reshaped to (%d, %d, %d) "%((self.framecount+idx), self._h, self._w, self._c))


        if self.out_type == "numpy":
            if "float" in self.dtype:
                _data = _data/self._255
                if self.timer is not None:
                    self.timer.subtic("to dtype %s, [%d]"%(self.dtype, self.framecount))

            self.data[idx] = _data
            if self.timer is not None:
                self.timer.subtic("fill np array at index %d, [%d]"%(idx, self.framecount))
        else:
            self.data[idx] = ((torch.from_numpy(_data).to(device=self.device).permute(2, 0, 1)).to(dtype=torch.__dict__[self.dtype])/self._div).contiguous()

            #self.data[idx] = torch.from_numpy((_data)).permute(2, 0, 1)
            if self.timer is not None:
                self.timer.subtic("[%d] fill tensor and permuted at index"%(idx+self.framecount))


        """
        data = np.frombuffer(self._pipe.stdout.read(self._bufsize), dtype=np.uint8).reshape(1, self._c, self._h, self._w)
        if self.timer is not None:
            self.timer.subtic("subtic, from buffer [%d]"%self.framecount)
        data = (torch.from_numpy(data).to(device=self.device).permute(0, 3, 1, 2).to(dtype=self.dtype)/self._div).contiguous()
        if self.timer is not None:
            self.timer.tic("to torch [%d]"%self.framecount)
        self.framecount += 1
        """


    def _init_data(self):
        if self.out_type == "numpy":
            return np.zeros([self.batch_size, self._h, self._w, self._c], self.dtype)
        else: # out_type: torch
            return torch.zeros([self.batch_size, self._c, self._h, self._w],
                               dtype=torch.__dict__[self.dtype], device=self.device, requires_grad=self.grad)
 
    def _build_cmd(self):

        self._cmd = [self._ffmpeg]

        # approximate search
        if self.start is not None:
            self._cmd += ["-ss", strftime(self.start-1)]

        self._cmd += ["-i", self.src]
        # precise search
        if self.start is not None:
            self._cmd += ["-ss", strftime(self.start)]

        #clip duration
        if self.frames is not None:
            self._cmd += ['-vframes', str(self.frames)]

        self._cmd += ["-f", "image2pipe", "-pix_fmt", self.pix_fmt, "-vcodec", "rawvideo"]
        self._cmd += ["-"]

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

    def _validate_batches(self):
        assert self.frames >= self.batch_size, "%srequesting batch size (%d) larger than number of frames in video: (%d)%s"%(Col.RB, self.batch_size, self.frames, Col.AU)
        _frames = self.batch_size*(self.frames//self.batch_size)
        if _frames < self.frames:
            print("%sskipping last %d frames from batch%s"%(Col.YB, (self.frames - _frames), Col.AU))
            self.frames = _frames

