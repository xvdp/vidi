"""
pipe to numpy
"""
import platform
import subprocess as sp
import numpy as np
import torch

from .utils import *
from .ff_fun import ffprobe

class FFread:

    def __init__(self, src, batch_size=1, out_type="numpy", start=None, frames=None, pix_fmt="rgb24", debug=False,
                 dtype="float", device="cpu", grad=False):
        """
        Args
            src      (str) valid video file
            start    (number [None])
                if float: interpret as time
                if int: interpret as frame
            frames (number, [None])
                if float: interpret as time
                if int: interpret as frame
            pix_fmt ()
        

        Examples:
            with vidi.FFcap(name+ext, pix_fmt=pix_fmt, size=size, overwrite=True, debug=True) as F:
        F.open_pipe()
        F.read(batch)

        
        F.stats: {"index", "codec_name", "codec_type", "width", "height", "pix_fmt",
                  "avg_frame_rate", "start_time", "duration", "nb_frames"}
        """
        self.src = src
        self.stats = ffprobe(src)

        self.start = self._as_time(start)
        self.frames = self.stats["nb_frames"] if frames is None else self._as_frame(frames)

        # TODO, pix_fmt handle
        self.pix_fmt = "rgb24" # yuv420p
        self._channels = 3
        if self.stats['pix_fmt'] == "gray":
            self._channels = "1"
            self.pix_fmt = "gray"

        self.batch_size = batch_size
        self.out_type = out_type
        self.dtype = dtype
        self.device = device
        self.grad = grad


        self._bufsize = self.stats["height"] * self.stats["width"] * self._channels
        self.data = self._init_data()

        self.debug = debug
        self._cmd = None
        self._pipe = None
        self._ffmpeg = 'ffmpeg' if platform.system() != 'Windows' else 'ffmpeg.exe'
        self._framecount = 0

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
        if self._cmd is None:
            self._build_cmd()
        if self._pipe is not None:
            self.close()
        self._pipe = sp.Popen(self._cmd, stdout=sp.PIPE, bufsize=self._bufsize)

    def close(self):
        if self._pipe is not None and not self._pipe.stdout.closed:
            self._pipe.stdout.flush()
            self._pipe.stdout.close()
        self._pipe = None

    def get_batch(self):
        self.data *= 0
        self._framecount += self.batch_size
        if self._framecount >= self.frames:
            return self.close()

        for i in range(self.batch_size):
            img_raw = self._pipe.stdout.read(self._bufsize) #self.stats["height"]*self.stats["width"]*self._channels)
            img = np.frombuffer(img_raw, dtype=np.uint8)
            self._update_data(i, img)

    def _update_data(self, idx, data):
        if self.out_type == "numpy":
            self.data[idx] = data.reshape(1, self.stats["height"], self.stats["width"], self._channels)
        else:
            self.data[idx] = torch.from_numpy(data.reshape(1, self._channels, self.stats["height"], self.stats["width"]))

    def _init_data(self):
        if self.out_type == "numpy":
            return np.zeros([self.batch_size, self.stats["width"], self.stats["height"], self._channels], self.dtype)
        else: # out_type: torch
            return torch.zeros([self.batch_size, self._channels, self.stats["width"], self.stats["height"]],
                               dtype=torch.__dict__[self.dtype], device=self.device, requires_grad=self.grad)
    """
    print(Col.YB, ' '.join(fcmd), Col.AU)
    #TODO
    _TODO_calculate_buffer_size = _width*_height*_channels*8

    try:
        T = Timer()
        pipe = sp.Popen(fcmd, stdout=sp.PIPE, bufsize=_TODO_calculate_buffer_size)
        T.tic("make pipe")

        for i in range(num_frames):
        #while True:
            #try:
            img_raw = pipe.stdout.read(_height*_width*_channels)

            #T.tic("read pipe [%d]"%i)
            #print(Col.YB, "type", type(img_raw), Col.AU)


            img = np.frombuffer(img_raw, dtype=np.uint8).reshape(_height, _width, _channels)
            #img = Image.frombuffer('RGB', (_width, _height), img_raw, "raw", 'RGB', 0, 1)

            # no scale: 68us
            # scale with PIL 640x480 * 2: 1.3 ms
            # scale with ffmpeg 640x480 * default : 6.8ms 
            # scale with ffmpeg 640x480 * 2 bilinear : 3.8ms 
            if scale != 1:
                img = Image.fromarray(img, 'RGB')
                img.resize(size)
            

            #print(Col.BB, "shape", img.shape, Col.AU)
            T.tic("to numpy [%d]"%i)

        cont = False

        ## tet stats
        pipe.stdout.flush()
    except:
        print(Col.RB, "CANT OPEN PIPE", Col.AU)

    # if pipe is not None:
    #     pipe.stdout.close()

    T.toc('flush')

    plt.imshow(img)
    plt.show()

    """
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
        assert False, "expected int (frames) or float (time), got "+type(value)

    def _as_time(self, value):
        """int or float to time"""
        if value is None or isinstance(value, float):
            return value
        if isinstance(value, int): # interpret as frame, return time
            return frame_to_time(value, fps=self.stats["avg_frame_rate"])
        assert False, "expected int (frames) or float (time), got "+type(value)