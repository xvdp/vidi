""" @xvdp
AVDataset legacy. todo validate against ffdb


"""
from typing import Union, Optional
import platform
import subprocess as sp
import numpy as np

from .utils import dprint, Col, frame_to_time, strftime, time_to_frame
from .ff import FF

_DEBUG = True


# pylint: disable=no-member
# pylint: disable=suppressed-message

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset

    class AVDataset(Dataset):
        def __init__(self,
                    src: str,
                    start: Union[int, float, None] = None,
                    frames: Union[int, float, None] = None,
                    step: int = 1,
                    ftransform: Optional[tuple] = None,
                    ttransform=None,
                    dtype=torch.float32,
                    device="cpu",
                    debug=False,
                    **kwargs) -> None:
            """
            Args
                src     (str)           ffmpeg readable video source
                start   (int [None])   start frame to start reading
                        (float [None]) start time (seconds)
                frames  (int [None])   number of frames to be read
                        (float [None]) time to be read
                step    (int [1]) step search between frames

                ftransform (list [None])   ffmpeg transforms  any filter in ffmpeg
                ttransform (object subclass [None])   torch; same syntax as torchvision transforms
                dtype (torch type ['float32])
                device  (str ['cpu']) cpu|cuda
                debug   (bool[False])

            Examples:
                ftransform = ftransform=["edgedetect=high=0", "negate"]     # getated edges
                ftransform = ftransform=["edgedetect=mode=colormix:high=0"] # cartoon colorization
                ftransform = ftransform=["edgedetect=mode=canny:low=0.3:high=0.5"] # canny
            """
            self.debug = debug

            self.src = src
            self.stats = FF(src).stats

            self.start = self._as_time(start)
            self.frames = self.stats["nb_frames"] if frames is None else self._as_frame(frames)
            self.step = step

            self.pix_fmt = kwargs.get('pix_fmt', "rgb24")
            self.dtype = dtype
            self.device = device

            self.ftransform = ftransform
            self.ttransform = ttransform
            self.framecount = 0

            self._c = 1 if self.stats['pix_fmt'] == "gray" else 3
            self._h = self.stats["height"]
            self._w = self.stats["width"]
            _resize = self.parse_ftransform(ftransform)
            if _resize:
                self._h = _resize['h']
                self._w = _resize['w']

            # size of buffer to retrieve per frame
            self._bufsize = self._h * self._w * self._c

            self._pipe = None
            self._ffmpeg = 'ffmpeg' if platform.system() != 'Windows' else 'ffmpeg.exe'
            self._cmd = None

        def __enter__(self):
            self.open()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()

        def __len__(self) -> int:
            return self.frames

        def __getitem__(self, i: int = 1) -> Optional[Tensor]:
            """ i can be used for ad hoc seek
                reads from buffer
                applies pil transforms
                applies numpy transforms
                applies torch transforms
            """
            if self._pipe is None or i is None or not i:
                return None

            # read ffmepeg frame

            for _ in range(i):
                data = self._pipe.stdout.read(self._bufsize)

            data = np.frombuffer(data, dtype=np.uint8).reshape(1, self._h, self._w, self._c)

            # build tensor #1, 640, 3, 480]
            data = torch.from_numpy(data).to(device=self.device,
                                            dtype=self.dtype).permute(0,3,1,2).contiguous()
            if self.dtype in (torch.float16, torch.float32, torch.float64):
                data.div_(255.0)

            # torch transforms
            if self.ttransform is not None:
                data = self.ttransform(data)

            self.framecount += 1
            return data

        def open(self) -> None:
            """ manual overwrite to with ... __enter__
            """
            if self._pipe is not None:
                self.close()
            if self._cmd is None:
                self._build_cmd()
            self._pipe = sp.Popen(self._cmd, stdout=sp.PIPE, bufsize=self._bufsize)


        def close(self) -> None:
            """ manual overwrite to with ... __exit__
            """
            if self._pipe is not None and not self._pipe.stdout.closed:
                self._pipe.stdout.flush()
                self._pipe.stdout.close()
                self._pipe.terminate()
            if self._pipe is not None and not self._pipe.stdout.closed:
                sp.run(['pkill', '-x', 'ffmpeg'], check=True) # this looks bad
            self._pipe = None


        def _build_cmd(self):
            self._cmd = [self._ffmpeg]

            #input
            self._cmd += ["-i", self.src]

            # precise search
            if self.start is not None and self.start >= 1:
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
                _step = f"framestep={self.step}"
                if _filters:
                    _filters = _step + "," + _filters
                else:
                    _filters = _step

            if _filters:
                self._cmd += ["-vf", _filters]

            self._cmd += ["-f", "image2pipe", "-pix_fmt", self.pix_fmt, "-vcodec", "rawvideo"]
            self._cmd += ["-"]
            if self.debug:
                dprint(f"{Col.GB} {self._cmd}, {Col.AU}", debug=self.debug)

        def _as_frame(self, value):
            """int or float to int frame"""
            if value is None or isinstance(value, int):
                return value
            if isinstance(value, float): # interpret as time, return frame
                return time_to_frame(value, fps=self.stats["rate"])
            assert False, f"{Col.RB}expected int (frames) or float (time), got {type(value)}{Col.AU}"

        def _as_time(self, value: Union[int, float, str, None]) -> float:
            """int or float to time"""
            if value is None or isinstance(value, float):
                return value
            if isinstance(value, int): # interpret as frame, return time
                return frame_to_time(value, fps=self.stats["rate"])
            elif isinstance(value, str):
                return sum(x * float(t) for x, t in zip([3600, 60, 1], value.split(":")))
            assert False, f"{Col.RB}expected int (frames) or float (time), got {type(value)}{Col.AU}"

        def parse_ftransform(self, ftransform: Optional[tuple]) -> Optional[dict]:
            """
            Any transformation that modifies buffer size,
            currently supported scale=, crop=,
                syntax supported: 'w=100:h=100:x=12:y=34' and '100:100'

                NOT supported: 'in_w' 'out_w' or maths ops. fix parse transform for that
                https://ffmpeg.org/ffmpeg-filters.html#Examples-50

            """
            dprint(f'{Col.GB}parse_ftransform { Col.YB}{ftransform}{Col.AU}', debug=self.debug)

            if not ftransform or ftransform is None:
                return None
            _bufmod = ['scale=', 'crop=']
            size = {'w':None, 'h':None}

            for _b in _bufmod:
                for _filter in ftransform:
                    if _b  in _filter:
                        _filter = _filter.split(_b)[1].split(":")

                        for i, _s in enumerate(_filter):
                            if "=" in _s:
                                _ss = _s.split("=")
                                size[_ss[0][0]] = int(_ss[1])
                            else:
                                size[list(size.keys())[i]] = int(_s)

                        if size['w'] is not None and size['h'] is not None:
                            return size
            return None

except: # pylint: disable=bare-except
    class AVDataset:
        def __init__(self):
            raise ModuleNotFoundError('torch not installed, AVDataset cannot be used')


    # def zoom(src, factor=1, center=(0.5, 0.5)):
    #     """ concatenate crop and scale for zoom
    #     Args:
    #         src (str video file)
    #         z   (float zoom factor)
    #         center (tuple [0.5,0.5]) in range 0 -1
    #     """
    #     stats = ffprobe(src)
    #     size = np.array([stats['width'], stats['height']])
    #     center = np.clip(np.array(center), 0, 1)
    #     _z = (size/factor).astype(int)
    #     offset = ((size - _z)* center).astype(int)
    #     return f"crop=w={_z[0]}:h={_z[1]}:x={offset[0]}:y={offset[1]},scale=w={size[0]}:h={size[1]}"

    # ffmpeg transforms, any transform valid in ffmpeg can be passed to this as a list in the ffmpeg syntax as per https://ffmpeg.org/ffmpeg-filters.html
    # ffmpeg transforms are executed globally or temporally as per ffmpeg syntax
    # Examples of temporal transforms also possible but not listed here

    # Examples of per frame transforms:
    #     ftransform = ["edgedetect=low=0.1:high=0.05"]
    #         = ["edgedetect=low=0.1:high=0.05", "negate"]
    #         = ["edgedetect=mode=colormix:high=0.7", "negate"]
    #         = ["edgedetect=mode=canny:high=0"]
    #     # histogram equalization
    #         = ["histeq=strength=0.5"]
    #         = ["histeq=strength=0.2:intensity=.9"]
    #     # lens distorts
    #         = ["lenscorrection=cx=0.5:cy=0.5:k1=-0.7:k2=-0.7"]
    #         = ['lenscorrection=cx=0.5:cy=0.5:k1=0.7:k2=0.7']
    #     # color balance
    #         = ["colorbalance=rs=.3"] # add red shift to shadows
    #         = ["colorbalance=gh=.5"] # add green shift to highlights
    #         = ["colorbalance=gm=.5:bm=.5"] # blue and red to midtobnes
    #     # channel mixer
    #         = ["colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131"] # simulate sepia
    #     # color levels
    #         = ["colorlevels=rimin=0.039:gimin=0.039:bimin=0.039:rimax=0.96:gimax=0.96:bimax=0.96"] # increase contrast
    #         = ["colorlevels=romin=0.5:gomin=0.5:bomin=0.5"] # increase brightness
    #     # convolution
    #         = ["convolution='1 1 1 1 -8 1 1 1 1:1 1 1 1 -8 1 1 1 1:1 1 1 1 -8 1 1 1 1:1 1 1 1 -8 1 1 1 1:5:5:5:1:0:128:128:0'"] # laplacian edge detector
    #     # curves
    #         = ["curves=r='0/0.11 .42/.51 1/0.95':g='0/0 0.50/0.48 1/1':b='0/0.22 .49/.44 1/0.8'"] # vintage effect
    #     # blur
    #         = ["gblur=sigma=6:sigmaV=0.1"]
    #     # gradient function
    #         = ['gradfun=radius=8'] # fix banding
    #     # white balance
    #         = ["greyedge=difford=1:minknorm=5:sigma=2"]
    #     # flip
    #         = ["greyedge=difford=1:minknorm=5:sigma=2"]
    #     # rotate
    #         = ["rotate=PI/6"]
    #     # zoom  #uses auxiliary function def zoom()
    #         = [zoom(src, z=2, center=(0,1))]
