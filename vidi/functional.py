"""@xvdp

Formats and FourCC <-> RGB
    get_formats()   # supported formats
    check_format()
    yxx2rgb()   conversion
    rgb2yxx()
    yxx_matrix()
    expand_fourcc() 4nm -> 444


General IO
    read_frame()
    read_bytes()
    to_bits()
    from_bits()
    images_to_video()
"""
import warnings
from typing import Union, Optional
import os
import os.path as osp
import subprocess as sp
import numpy as np
import cv2

from .utils import frame_to_strftime

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=suppressed-message

def _make_subtitles(name: str, nb_frames: int, fps: float,  start_frame: int = 0) -> str:
    """ Make a subtitle file with frame numbers and timestamps
    as standalone gist https://gist.github.com/xvdp/ec64daaa6fa381d8ba02342801a51b37
    return name of subtitle file

    Args:
        name        (str[None]) output name, default VideoName.srt
        start_frame (int [0]) ffmpeg exports default ot 1

    # to load in ffplay
    ffplay <video_fname> -vf subtitles=<subtitle_name>
    # loads automatically in vlc provided name suffix is the same
    vlc <video_fname>

    # srt syntax
    0                               # frame number: start_frame
    00:00:00.000 --> 00:00:00.033   # from to time
        0	00:00:00.000            # subtitle

    1                               # next subtitle ...
    00:00:00.033 --> 00:00:00.066
        1	00:00:00.033
    """
    name = name+'.srt' if '.srt' not in name else name
    sub = ""
    last_frame = frame_to_strftime(0, fps)
    for i in range(nb_frames):
        _frame = i + start_frame
        next_frame = frame_to_strftime(_frame+1, fps)
        sub += f"{_frame}\n{last_frame} --> {next_frame}\n\t{_frame}\t{last_frame}\n\n"
        last_frame = next_frame

    with open(name, 'w', encoding='utf8') as _fi:
        _fi.write(sub)
    return name


def read_frame(buffer: Union[str, bytes],
               pix_fmt: str,
               width: int,
               height: int,
               channel_axis = -1,
               interpolation: int = 1,
               compressed: bool = False,
               to_rgb: bool = False,
               dtype: Union[str, np.dtype, None] = np.float32) -> Union[np.ndarray, list]:
    """ bytes to ndarray ofor a single frame

    Args
        buffer  (str, bytes)    length of data needs to correspond to pix_fmt data packing
        pix_fmt (str) little endian pix_fmt supported by ffmpeg in (rgb, bgr)/a or yuv/a fourcc
            run .get_formats() to see currently implemented formats
        width   (int) expected width
        height  (int) expected out
        interploation (int) 0 NEAR, 1 LINEAR, 2 CUBIC 4 LANCZOS
        compressed      (bool [False]) for fourcc codecs, keep compression: returns list
    """
    buffer = read_bytes(buffer)
    bits, _, fourcc = check_format(pix_fmt)

    bitdepth = int(bits[0]) # will this be wrong for scaling bits?
    channels = len(bits)
    _dtype = np.uint8 if np.ceil(np.log2(bitdepth)) == 3 else np.uint16
    frame = np.frombuffer(buffer, dtype=_dtype)


    if fourcc is not None:
        shape = (channels, height, width) if channel_axis == 0 else (height, width, channels)
        out =  expand_fourcc(frame, fourcc, shape, interpolation, channel_axis, compressed)
    else:
        out = []
        _a0 = width * height
        for i in range(channels):
            out.append(frame[i*_a0:(i+1)*_a0].reshape(height, width))
    if not compressed or all([out[0].shape == _a.shape for _a in out[1:]]):
        out = np.stack(out, axis=channel_axis)

    # convert to float in (0,1) range
    if dtype is not None or (to_rgb and 'yuv' in pix_fmt):
        dtype = dtype or np.float32
        out = from_bits(out, int(bits[0]), dtype=dtype)

    if isinstance(out, np.ndarray) and (to_rgb and 'yuv' in pix_fmt):
        if 'ayuv' in pix_fmt:
            out[..., 1:] = yxx2rgb(out[..., 1:], clamp=True)
        elif 'yuva' in pix_fmt:
            out[..., :-1] = yxx2rgb(out[..., :-1], clamp=True)
        else:
            out = yxx2rgb(out, clamp=True)

    return out


def rgb_to_yuv(frame: np.ndarray,
               pix_fmt: str,
               compress: bool = True,
               interpolation: int = 1) -> Union[list, np.ndarray]:
    """if compress: returns list, otherwise ndarray, both IO are floats
    Args
        frame           np.ndarray ndim 3, float
        pix_fmt
        compress        (bool [True])   # yuv compression 422 etc
        interpolation   (int) 0 NEAR, 1 LINEAR, 2 CUBIC 4 LANCZOs:
    """
    assert 'yuv' in pix_fmt

    bits, _, fourcc = check_format(pix_fmt)

    channels = len(bits)
    channel_axis = _get_channels_axis(frame.shape, channels)
    out = frame.copy()

    if channel_axis == 0:
        out = out.transpose(1,2,0) # mm is simpler as HWC
    _s = slice(1 if 'ayuv' in pix_fmt else 0, -1 if 'yuva' in pix_fmt else None)
    out[..., _s] = rgb2yxx(out[..., _s], mode='YUV_RGB_709', clamp=True)

    if compress:
        out =  compress_fourcc(out, fourcc, interpolation)
    return out


def write_frame(frame: np.ndarray,
                pix_fmt: str,
                from_rgb: bool = False,
                compress: bool = True,
                interpolation: int = 1) -> bytes:
    """  (HWC | CHW) ndarray to flat buffer
    Converts RGB to YUV if pix_fmt is yuv and from_rgb == True
    

    Args
        frame   (np.ndarray) HWC or CHW, float( 0-1)
        pix_fmt (str) little endian pix_fmt supported by ffmpeg in (rgb, bgr)/a or yuv/a fourcc
            run .get_formats() to see currently implemented formats
        from_rgb        (bool [False]) # if pix_fmt is yuv- and True, convert channels
        compress        (bool [True])   # yuv compression
        interploation   (int) 0 NEAR, 1 LINEAR, 2 CUBIC 4 LANCZOS: for yuv compression

    """
    bits, _, fourcc = check_format(pix_fmt)

    # convert to yuvcompress
    if 'yuv' in pix_fmt and from_rgb:
        frame = rgb_to_yuv(frame, pix_fmt, compress, interpolation)
    elif 'yuv' in pix_fmt and compress:
        frame =  compress_fourcc(frame, fourcc, interpolation)

    if not isinstance(frame, list):
        channels = len(bits)
        channel_axis = _get_channels_axis(frame.shape, channels)
        frame = [np.squeeze(f, channel_axis) for f in np.split(frame, channels, axis=channel_axis)]

    # flat array
    out = np.concatenate([o.reshape(-1) for o in frame])

    # to buffer ready data
    bitdepth = int(bits[0])
    dtype = np.uint8 if np.ceil(np.log2(bitdepth)) == 3 else np.uint16
    return to_bits(out, bitdepth, dtype=dtype).tobytes()


def from_bits(x: Union[list[np.ndarray], np.ndarray],
              bits: int = 10,
              dtype: np.dtype = np.float32) -> np.ndarray:
    """ uilt to float
    Args
        x       (ndarray)
        bits    (int [10]) number of bits of input data
        dtype   (dtype [np.float32])
    """
    if isinstance(x, np.ndarray):
        return  (x/(2**bits-1)).astype(dtype)

    for i, y in enumerate(x):
        x[i] = from_bits(y, bits, dtype)
    return x


def to_bits(x: np.ndarray, bits: int = 10, dtype: np.dtype = np.uint16) -> np.ndarray:
    """ float to uint
    """
    return( x*(2**bits-1)).astype(dtype)


def read_bytes(data: Union[str, bytes]) -> bytes:
    """ read binary file or pass thru bytes
    Args
        data    (str | bytes)
    """
    if isinstance(data, str):
        assert osp.isfile(data), f'expected file or bytes, no file <{data}> found'
        with open(data, mode="rb") as _fi:
            data = _fi.read()
    assert isinstance(data, bytes), f'expected file or bytes got {type(data)}'
    return data

def get_size(size: Union[str, tuple, int]) -> Optional[str]:
    if isinstance(size, str) and size.isnumeric():
        size = int(size)
    if isinstance(size, int):
        size = (size, size)
    if isinstance(size, (tuple, list)):
        size = f"{size[0]}:{size[1]}"
    if isinstance(size, str) and len(size.split(':')) == 2:
        return size
    return None

def str_op(dic, key, val=None, prefix='-', conj=' ') -> str:
    if key in dic or val is not None:
        return f"{prefix}{key}{conj}{dic.get(key, val)}"
    return ""


def get_frame_rate(frame_rate: Union[float, str]) -> float:
    """ resolve frame rate from str
    """
    _strrates = {'ntsc': 30000/1001, 'pal': 25/1, 'qntsc': 30000/1001 , 'qpal': 25/1,
                 'sntsc': 30000/1001, 'spal': 25/1, 'film': 24/1 , 'ntsc-film': 24000/1001}
    if isinstance(frame_rate, str):
        frame_rate = _strrates[frame_rate]
    return frame_rate


def images_to_video(dst: str,
                    src: str,
                    start_number: Optional[int],
                    overwrite: bool = False, **kwargs) -> str:
    """ creates a video file from sequence of patterned images, default is prores yuv422p10le 
    Args
        dst             (str) output file               e.g. metropolis_color.mov
        src             (str) input patterned file      e.g. metro%08d.png
        overwrite       (bool [False])
        start_number    (int) source pattern start file_number # if concat enabled, make optional
    kwargs
        frame_rate      (float, str [24000/1001])  str in ('ntsc', 'pal', 'qntsc', 
                                    'qpal', 'sntsc', 'spal', 'film', 'ntsc-film')
        pix_fmt         (str ['yuv420p10le'])   in ffmpeg -pix_fmts | grep O
        scale           (int, tuple, str)   str as f'{width}:{height}'
        sar             (int) sample aspect ratio, if sar and no scale, scales appropriately
        vcodec          (str ['prores']) huffyuv, h264, mjpeg, mpeg4, h264, ... in ffmpeg -codecs
            profile     (str) ['hq'] # if vcodec == prores
        color_trc       (str ['bt709']) | gamma22 gama28 linear, log, ,...
        color_range     (str ['tv'])
        field_order     (str ['progressive']) | tt, bb, tb, bt
        time_base       (float [1/frame_rate])
        timecode        (str)   # 01:20:10:05
        vframes         (int)   # max number of frames to include

    >>> src = 'some_file_%03d.png'
    >>> cmd = images_to_video('TEST.mov', src, start_number=127, sar=2, overwrite=True, timecode="01:20:10:05")
    >>> cmd = images_to_video('TEST.mov', src, start_number=127, frame_rate=24000/1001, scale=(512,512), pix_fmt="yuvj420p", vcodec="mjpeg", overwrite=True, b="128K", vframes=10)
    >>> cmd = images_to_video(dst = "Merged.mov", src=f, start_number=170)
    """

    # output
    # dst = osp.abspath(osp.expanduser(dst))
    if osp.isfile(dst):
        if not overwrite:
            warnings.warn(f"No file written, {dst} exists, set overwrite=True")
            return None
        else:
            dst = f"-y {dst}"

    # frame rate
    frame_rate = get_frame_rate(kwargs.get('frame_rate',  24000/1001))
    start = 0

    if '%' in src:
        assert start_number is not None, f"start_number req' with patterned {src}"
        start = f"-start_number {start_number}"
    _test_src = src if '%' not in src else src%start_number
    assert osp.isfile(_test_src), f"file {_test_src} not found"
    src = f'-i {src}'

    pix_fmt = str_op(kwargs, 'pix_fmt',  'yuv422p10le')
    vcodec = str_op(kwargs, 'vcodec',  'prores_ks') # c:v'
    vframes = str_op(kwargs, 'vframes', prefix=' -') # max num of frames to include

    opts = [str_op(kwargs, 'color_range', 'tv'), # mpeg, pc, jpeg
               str_op(kwargs, 'field_order', 'progressive'), # tt, bb, tb, bt
               str_op(kwargs, 'time_base',  1/frame_rate)]

    if 'prores_ks' in vcodec:
        opts += [str_op(kwargs, 'profile', 'hq', conj=":v "),
                 str_op(kwargs, 'color_trc', 'bt709'),]

    scale = get_size(kwargs.get('scale'))
    sar = kwargs.get('sar')
    if sar:
        opts += [str_op(kwargs, 'sar')]
        if scale is None:
            opts += [f"-vf scale='trunc(iw/{sar}):ih'"]
    if scale is not None:
        opts += [f"-vf scale={scale}"]

    # TODO Add audio codec and options
    _opts = ['b',        # bits/sec [200K]   -b 1200K (8000K)
             'ab',       # audio bits/sec [128k]
             'flags',    # mv
             'ar',       # audio sampling Hz
             'ac',       # audio channels
            #  'b:a',      # audio bit rate 96k
             'timecode'
             ]      # sample aspect ratio

    opts += [f" -{op} {val}" for op, val in kwargs.items() if op in _opts]
    opts = " " + " ".join([o for o in opts if o])

    cmd = f'ffmpeg {start} {src}{vframes} -r {frame_rate} {pix_fmt} {vcodec}'
    cmd += f'{opts} {dst}'
    if kwargs.get('run', True):
        sp.call(cmd.split())
    return cmd


def get_formats(supported: bool = True, unsupported: bool = False) -> dict:
    """ return available ffmpeg patterns and if this reader supports them
    """
    _notin = ('bgr444', '555', '565', '_byte', 'bgr4', 'rgb444', '410')
    _not = ('rgb4', 'bgr8', 'rgb8', 'rgb4')
    _prefixes = ('rgb', 'bgr', 'arg', 'abg', 'yuv', 'ayu', '0rg', '0bg')
    out = {}
    with os.popen("ffprobe -v quiet -pix_fmts") as _fi:
        _fmts = _fi.read().split("\n")
    for _, _line in enumerate(_fmts[8:]):
        _fmt = _line.split()
        if len(_fmt) >= 4 and _fmt[2].isnumeric() and  _fmt[3].isnumeric():
            _issupported = False
            if not _fmt[1].endswith('be') and _fmt[1][:3] in _prefixes and \
                _fmt[1] not in _not and all(n not in _fmt[1] for n in _notin):
                _issupported = True
            if ((_issupported and supported) or (not _issupported and unsupported)):
                out[_fmt[1]] = {'flags':_fmt[0].replace('.', ''),
                                'channels': int(_fmt[2]),
                                'bpp': int(_fmt[3])}
                if supported and unsupported:
                    out[_fmt[1]]['supported'] = _issupported
    return out


def get_encoders() -> list:
    """ return available ffmpeg patterns and if this reader supports them
    """
    video = []
    audio = []
    with os.popen("ffprobe -v quiet -encoders") as _fi:
        _encs = _fi.read().split("\n")
    for i, _enc in enumerate(_encs[10:]):
        _test =  _enc.replace(' ','')
        if _test and _test[0] == 'V':
            video.append(_enc.split()[1])
        elif _test and _test[0] == 'A':
            audio.append(_enc.split()[1])
    return video, audio

def check_video_encoder(vcodec: str) -> bool:
    """ validates that requested encoder is in ffmpeg version
    """
    return vcodec in get_encoders()[0]

def check_audio_encoder(acodec: str) -> bool:
    """ validates that requested encoder is in ffmpeg version
    """
    return acodec in get_encoders()[1]

def check_format(pix_fmt: str) -> tuple:
    """ return list of channels with bits
    only littleendian formats supported
    # TODO simplify fourcc parsing & add alpha channel position

    A bit redundant except for the yuv compression: ffprobe -pix_fmts returns info.
    """
    _supported = get_formats()
    assert pix_fmt in _supported, f"{pix_fmt} unsupported, use {sorted(list(_supported.keys()))}"

    out = None
    order = None
    fourcc = None

    _pix_fmt = pix_fmt
    if _pix_fmt.endswith('le'):
        _pix_fmt = _pix_fmt[:-2]

    if '64' in _pix_fmt:
        out = [16,16,16,16]
        order = _pix_fmt.split('64')[0]
    elif '48' in _pix_fmt:
        out = [16,16,16]
        order = _pix_fmt.split('48')[0]
    elif _pix_fmt in ('rgb24', 'rgb0', '0rgb', 'bgr24', 'bgr0', '0bgr'):
        order = 'rgb' if 'rgb' in _pix_fmt else 'bgr'
        out = [8,8,8]
    elif _pix_fmt in ('bgra', 'rgba', 'abgr', 'argb'):
        order = _pix_fmt
        out = [8,8,8,8]

    elif _pix_fmt[:3] == 'yuv':
        order = 'yuv'
        bits = 8
        components = 3
        j = 1
        if _pix_fmt[-j].isnumeric():
            while _pix_fmt[-j].isnumeric():
                j += 1
            bits = int(_pix_fmt[-j+1:])
        _pix_fmt = _pix_fmt.split('yuv')[1]
        if not _pix_fmt[0].isnumeric():
            if _pix_fmt[0] == 'a':
                order += 'a'
                components += 1
            _pix_fmt = _pix_fmt[1:]
        if not _pix_fmt[:3].isnumeric():
            raise NotImplementedError(f"format not implemented {pix_fmt}")
        out = [bits*int(c)/4 for c in _pix_fmt[:3]]
        fourcc = [int(c) for c in _pix_fmt[:3]]
        if components == 4:
            out = out + [bits]
            fourcc = fourcc + [fourcc[0]]

    return out, order, fourcc # this could be simpler fourcc == out*bits/4


###
#
# ycc and yuv <-> rgb
#
def yxx_matrix(mode: str, invert: bool = False, dtype: np.dtype = np.float32) -> np.ndarray:
    """ color conversion matrices
    Args:
        mode    (str) in (YUV_RGB_709, YCC_RGB_709, YCC_RGB_2020)
        invert  (bool [False]) return inverse matrix
    """
    mat = {}
    mat['YUV_RGB_709'] = np.array([[ 1.     ,  1.     ,  1.     ],
                                   [ 0.     , -0.39465,  2.03211],
                                   [ 1.13983, -0.5806 ,  0.     ]], dtype=dtype)
    mat['YCC_RGB_709'] = np.array([[ 1.    ,  1.    ,  1.    ],
                                   [ 0.    , -0.1873,  1.8556],
                                   [ 1.5748, -0.4681,  0.    ]], dtype=dtype)
    mat['YCC_RGB_2020'] = np.array([[ 1.        ,  1.        ,  1.        ],
                                    [ 0.        , -0.16455313,  1.8814    ],
                                    [ 1.4746    , -0.57135313,  0.        ]], dtype=dtype)

    assert mode in mat, f"got {mode}, expected {list(mat.keys())}"
    out = mat[mode]
    if invert:
        out = np.linalg.inv(out)
    return out

def rgb2yxx(rgb: np.ndarray, mode: str = 'YUV_RGB_709', clamp: bool = False) -> np.ndarray:
    """ float rgb to yuv444 or ycc444
    Args
        rgb     (ndarray) shape (h,w,3) float
        mode    (str [YUV_RGB_709]) color profile
            (YUV_RGB_709, YCC_RGB_709, YCC_RGB_2020)
        clamp   (bool [False]) clamp result to 0,1
    """
    mat = yxx_matrix(mode, True, dtype=rgb.dtype)
    shape = rgb.shape
    out = (rgb.reshape(-1,3) @ mat).reshape(shape)
    out[..., 1] += 0.5
    out[..., 2] += 0.5
    if clamp:
        out = np.clip(out, 0, 1)
    return out


def yxx2rgb(yuv: np.ndarray, mode: str = 'YUV_RGB_709', clamp: bool = False) -> np.ndarray:
    """ float yuv444 or ycc444 to rgb
    Args
        yuv     (ndarray) shape (h,w,3) float
        mode    (str [YUV_RGB_709]) color profile
            (YUV_RGB_709, YCC_RGB_709, YCC_RGB_2020)
        clamp   (bool [False]) clamp result to 0,1
    """
    mat = yxx_matrix(mode, False, dtype=yuv.dtype)
    shape = yuv.shape
    out = yuv.copy()
    out[..., 1] -= 0.5
    out[..., 2] -= 0.5
    out = (out.reshape(-1,3) @ mat).reshape(shape)
    if clamp:
        out = np.clip(out, 0, 1)
    return out


def expand_fourcc(frame: np.ndarray,
                  fourcc: tuple,
                  shape: tuple,
                  interpolation: int = 1,
                  channel_axis: int = -1,
                  compressed: bool = False) -> Union[np.ndarray, list]:
    """ flat ndarray yuv | yuva 444, 440, 422, 420, 410 to stacked 444 ndarray
    Args
        frame   (ndarray) flat
        fourcc  (tuple ) # four cc code expanded to include alpha
        shape   (tuple) # shape of output array
        interpolation   (int [1]) | 0: nearest | 1:linear | 2:cubic | 4:lanczos4
        channel_axis    (int [-1]) | 0 :  CHW or HWC
        compressed      (bool [False]) for fourcc codecs, keep compression: returns list
    """
    out = []
    shape = list(shape)
    channels = shape.pop(channel_axis)
    height, width = shape

    _area_ratio = sum(fourcc[1:3])/(fourcc[0]*2)
    _width_ratio = fourcc[1]/fourcc[0]
    _height_ratio = _area_ratio/_width_ratio

    _area = width * height
    _area_uv = int(_area_ratio * _area)
    _width_uv = int(_width_ratio * width)
    _height_uv =  int(_height_ratio * height)
    _address = np.cumsum([_area, _area_uv, _area_uv])

    out.append(frame[:_area].reshape(height, width))
    for i in range(1,3):
        data = frame[_address[i-1]:_address[i]].reshape(_height_uv, _width_uv)
        if _area_ratio != 1 and not compressed:
            data = cv2.resize(data, dsize=(width, height), interpolation=interpolation)
        out.append(data)

    if channels == 4:
        out.append(frame[_address[-1]:].reshape(height, width))
    return out

def _get_channels_axis(shape: tuple, num_channels: int) -> int:
    channel_axis = -1 if shape[-1] == num_channels else 0
    assert shape[channel_axis] == num_channels, \
        f"neither dim 0 nor -1 has {num_channels} channels: {shape}"
    return channel_axis


def compress_fourcc(frame: np.ndarray,
                    fourcc: tuple,
                    interpolation: int = 1) -> list[np.ndarray]:
    """ stacked 444 ndarray to  flat ndarray yuv | yuva 444, 440, 422, 420, 410
    Args
        frame   (ndarray) stacked ndim =  3
        fourcc  (tuple ) # four cc code expanded to include alpha
        interpolation   (int [1]) | 0: nearest | 1:linear | 2:cubic | 4:lanczos4
    """
    assert frame.ndim == 3, f"expected HWC or CHW, got {frame.shape}"
    channel_axis = _get_channels_axis(frame.shape, len(fourcc))

    shape = list(frame.shape)
    channels = shape.pop(channel_axis)
    height, width = shape

    data = [np.squeeze(f, channel_axis) for f in  np.split(frame, channels, axis=channel_axis)]

    _area_ratio = sum(fourcc[1:3])/(fourcc[0]*2)
    if _area_ratio < 1:
        _width_ratio = fourcc[1]/fourcc[0]
        _height_ratio = _area_ratio/_width_ratio
        _width_uv = int(_width_ratio * width)
        _height_uv =  int(_height_ratio * height)

        for i, d in enumerate(data):
            if i in (1, 2):
                data[i] = cv2.resize(data[i], dsize=(_width_uv, _height_uv),
                                    interpolation=interpolation)

    return data
