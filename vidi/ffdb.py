"""@xvdp
ffmpeg dataset to torch/ wip


    with FFDataset(videofile) as D:
        D.__getitem__() # -> torch tensor
        D.__len__()
"""
from typing import Union, Optional, Callable
import subprocess as sp
import warnings
import numpy as np
from .ff import FF
from .functional import check_format, read_frame


# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=suppressed-message
_WITH_TORCH = True
try:
    import torch
    from torch import Tensor
except: # pylint: disable=bare-except
    _WITH_TORCH = False
    Tensor = np.ndarray
    warnings.warn("torch not found, FFDataset will output ndarrays")


class FFDataset(FF):
    """ WIP Dataset to output ffmpeg to torch tensors
    """
    def __init__(self,
                 fname: str,
                 start: Union[int, float, str] = 0,
                 nb_frames: Optional[int] = None,
                 end: Union[int, float, str, None] = None,
                 scale: Union[float, tuple] = 1.,
                 scale_aspect_ratio: int = 0,
                 crop: Optional[tuple[int,int,int,int]] = None,
                 transforms: Optional[Callable] = None,
                 compressed: bool = False,
                 device: str = 'cpu',
                 to_rgb: bool = False,
                 **kwargs) -> None:
        super().__init__(fname)

        self._cmd = None
        self._pipe = None
        self._bufsize = None

        self._start =start
        self._nb_frames = nb_frames
        self._end = end
        self.scale = scale
        self.scale_aspect_ratio = scale_aspect_ratio
        self.crop = crop
        self.compressed = compressed
        self.to_rgb = to_rgb

        self.ttransform = transforms
        self.framecount = 0
        self.device = device
        _dtype = torch.get_default_dtype() if _WITH_TORCH else np.float32
        self.dtype = kwargs.get('dtype', _dtype)

        self.pix_fmt = self._out_format(**kwargs)
        self.build_cmd()


    def __enter__(self):
        self.open()
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


    def __len__(self) -> int:
        return self.frame_range[1]


    def __getitem__(self, i: int = 1) -> Optional[Tensor]:
        """ i can be used for ad hoc seek
            reads from buffer
            applies pil transforms
            applies numpy transforms
            applies torch transforms
        """
        if self._pipe is None or i is None or not i:
            return None
        buffer = self._pipe.stdout.read(self._bufsize)

        channel_axis = 0
        _transpose = False
        if self.to_rgb and 'yuv' in self.pix_fmt:
            channel_axis = -1
            _transpose = True

        # possible TODO: read frame as tensor directly
        data = read_frame(buffer, self.pix_fmt, self.stats['out_width'],
                          self.stats['out_height'], channel_axis=channel_axis,
                          interpolation=1, compressed=self.compressed,
                          to_rgb=self.to_rgb, dtype=np.float32)[None]


        if _WITH_TORCH:
            if isinstance(data, np.ndarray):
                data = torch.as_tensor(data, device=self.device, dtype=self.dtype)
                if _transpose:
                    data = data.permute(0,3,1,2).contiguous()
                if self.ttransform is not None:
                    data = self.ttransform(data)
            else:
                data = [torch.as_tensor(d, device=self.device, dtype=self.dtype)
                        for d in data]

        self.framecount += 1
        if self.framecount >= self.frame_range[-1]:
            self.close()
        return data


    def open(self) -> None:
        """ manual overwrite to with ... __enter__
        """
        if self._pipe is not None:
            self.close()
        if self._cmd is None:
            self.build_cmd()
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
        print(f"Closing datast on {self.file} after {self.framecount} frames")
        self.framecount = 0


    def _out_format(self, **kwargs):
        """ auto expand format using ffmpeg
        """
        if 'pix_fmt' in kwargs:
            return kwargs['pix_fmt']
        pix_fmt = self.stats['pix_fmt']
        if not self.compressed:
            for _4cc in ('440', '422', '420', '411', '410'):
                if _4cc in pix_fmt:
                    pix_fmt = pix_fmt.replace(_4cc, '444')
                    break
        return pix_fmt


    def _process_kwargs(self, **kwargs):
        for key, val in kwargs.items():
            if key in ('pix_fmt', 'start', 'nb_frames', 'end', 'scale',
                       'scale_aspect_ratio', 'crop'):
                self.__dict__[key] = val


    def build_cmd(self, **kwargs):
        """ builds ffmpeg commant
        """
        self._process_kwargs(**kwargs)
        self._cmd = ['ffmpeg', "-i", self.file]
        self._cmd += self.format_start_end(self._start, self._nb_frames, self._end)
        self._cmd += self.build_filter_graph(scale=self.scale,
                                             scale_aspect_ratio=self.scale_aspect_ratio,
                                             crop=self.crop)
        self._cmd += ['-start_number', str(self.frame_range[0]),
                      "-f", "image2pipe", "-pix_fmt", self.pix_fmt, "-vcodec", "rawvideo", "-"]
                    #   '-f', 'rawvideo',
                    #   '-pix_fmt', self.pix_fmt, 'pipe:']

        bits, *_ = check_format(self.pix_fmt)
        self._bufsize = self._get_bufsize(bits, nb_frames=1, crop=self.crop)
