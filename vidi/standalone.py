""" @xvdp
standalone functions

    make_subtitles(videofile)       -> .str subtitles with frame info
    export_frames(videofile)        -> saved images
    export_clip(videofile)          -> videofile
    frames_to_numpy(videofile)      -> ndarray

"""
from typing import Optional, Union
import numpy as np

from .ff import FF

def make_subtitles(filename: str, subname: Optional[str] = None, start_frame: int = 0) -> str:
    """ makes an srt subtitle file that shows frame number and time string
    Args
        filename    (str) valid video file
        subname     (str [None]) subtitle file name, if None: filename.replace(<.ext>, '.srt')
    """
    vid = FF(filename)
    return vid.make_subtitles(name=subname, start_frame=start_frame)


def export_frames(filename: str,
                  start: Union[int, float] = 0,
                  nb_frames: Optional[int] = None,
                  end: Union[int, float, str, None] = None,
                  scale: Union[float, tuple[float, float]] = 1.,
                  stream: int = 0,
                  out_name: Optional[str] = None,
                  out_folder: str = None,
                  scale_aspect_ratio: int = -1,
                  **kwarg) -> str:
    """ extract frames from video

    # show Iframes 
    ffprobe -select_streams v -show_frames -show_entries frame=pict_type -of csv $N | grep -n I
    # play vid at time # time doesnt match subtitles
    ffplay -i $VIDNAME -ss 00:01:43.666 -t 10 -vf subtitles=$SUBTITLENAME #.srt

    Args
        out_name    (str)   # if no format in name ".png
        start       (int|float [0])  float: time in seconds, int: frame number
        nb_frames   (int [None]):   default: to end of clip
        end         (int, float, str, None) overrides nb_frames
        scale       (tuple float [1]) rescale output
        stream      (int [0]) if more than one stream in video
        out_folder  optional, save to folder
        scale_aspect_ratio  (int, [-1]), 0, 1: if 'sample_aspect_ratio' in clip and not 1:1
            default: -1: downscales extended dim; 1: up scales shorter dim; 0: no scaling

    kwargs
        out_format  (str) in (".png", ".jpg", ".bmp") format override
        crop        (list, tuple (w,h,x,y)
    """
    vid = FF(filename)
    return vid.export_frames(start, nb_frames, end, scale, stream, out_name, out_folder,
                             scale_aspect_ratio, **kwarg)

def export_clip(filename: str,
                out_name: Optional[str] = None,
                start: Union[int, float, str] = 0,
                nb_frames: Optional[int] = None,
                end: Union[int, float, str, None] = None,
                scale: Union[float, tuple[float, float]] = 1.,
                stream: int = 0,
                out_folder: str = None,
                overwrite: bool = False,
                **kwargs):
    """ extract video clip
    Args:
        out_name    (str [None]) default is <input_name>_framein_frame_out<input_format>
        start       (int | float [0])  default: start of input clip, if float: time
        nb_frames   (int [None]):      default: to end of clip
        end         (int, float, str, None) overrides nb_frames
        scale       (float [1]) rescale output
        stream      (int [0]) if more than one stream in video
    kwargs
        out_format  (str) in (".mov", ".mp4") format override
        crop        (list, tuple (w,h,x,y)
    ffmpeg -i a.mp4 -force_key_frames 00:00:09,00:00:12 out.mp4
    """
    vid = FF(filename)
    return vid.export_clip(out_name, start, nb_frames, end, scale, stream, out_folder,
                           overwrite, **kwargs)


def frames_to_numpy(filename: str,
                    start: Union[int, float, str] = 0,
                    nb_frames: Optional[int] = None,
                    end: Union[int, float, str, None] = None,
                    scale: Union[float, tuple] = 1.,
                    stream: int = 0,
                    scale_aspect_ratio: int = -1,
                    to_rgb: bool = False,
                    crop: Optional[tuple[int,int,int,int]] = None,
                    compressed: bool = False,
                    dtype: Union[str, np.dtype, None] = np.float32,
                    **kwargs) -> np.ndarray:
    """
    read video to float numpy
        Args
            start               (int|float [0])  float: seconds, int: frame, str: HH:MM:SS.mmm
            nb_frames           (int [None]) None: all frames
            end                 (int, float, str, None) overrides nb_frames
            scale               (float [1])
            stream              (int [0]) video stream to load
            scale_aspect_ratio  (int [-1]) -1 downscale, 0 None, or upscale 1 aspect ratio ['sar']
            to_rgb              (bool [False]) if fourcc and True: converts
            crop                (tuple (w,h,x,y) [None])    -> crop=crop[0]:crop[1]}crop[2]:crop[3]
            compressed          (bool [False]) if fourcc and True: return list of arrays
            dtpye               (np.dtype [np.float32])
        kwargs
            pix_fmt             convert to pix_fmt before to_numpy()
            channel_axis        (int in -1, 0) -1: H,W,C  0: C,H,W
            interpolation       (int in 0,1,2,4) 0 NEAR, 1 LINEAR, 2 CUBIC 4 LANCZOS
    """
    vid = FF(filename)
    return vid.to_numpy(start, nb_frames, end, scale, stream, scale_aspect_ratio, to_rgb,
                        crop, compressed, dtype, **kwargs)
