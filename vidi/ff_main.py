"""@xvdp
File to handle ffmpeg requests
"""
from typing import Union, Optional
import warnings
import os
import os.path as osp
import subprocess as sp
import platform
import json
import numpy as np
import matplotlib.pyplot as plt


from .utils import Col, CPUse, GPUse
from .io_main import IO

class FF():
    """wrapper class to ffmpeg, ffprobe, ffplay
        Examples:
            >>> from vidi import FF
            >>> f = FF('MUCBCN.mp4')
            >>> print(f.stats)
            >>> c = f.clip(start=100, nb_frames=200) #output video clip
            # output scalled reformated video clip
            >>> d = f.clip(start=500, nb_frames=300, out_format=".webm", scale=0.25)
            # output images
            >>> e = f.get_frames(out_format='.png', start=100, nb_frames=5, scale=0.6)
            >>> f.play(c)


    v = FF(<myvideofile>)
    Image.fromarray(V.to_numpy(start='00:03:47.933',nb_frames=1)[0]).save('C_000347933'.png')
    Image.fromarray(V.to_numpy(start=6838, nb_frames=1)[0]).save('C_6838.png') 
        -> at fps=30. is identical  strftime(6838/30) == '00:03:47.933'
    V.export_frames(start='00:03:47.933', nb_frames=1)
    V.export_frames(start=6838, nb_frames=1)
    V.export_frames(start=6838, end=6839)
    V.export_frames(start=227.933, nb_frames=1)
         -> at fps=30. is identical

    V.export_clip(start='00:03:47.933', nb_frames=30, overwrite=True)
    V.export_clip(start='00:03:47.933', end='00:03:48.933', overwrite=True)
         -> at fps=30. is identical
    """
    def __init__(self, fname: Optional[str] = None) -> None:

        self.ffplay = 'ffplay'
        self.ffmpeg = 'ffmpeg'
        self.ffprobe = 'ffprobe'
        self.stats = {}
        self._io = IO()
        self.file = fname
        if fname is not None and osp.isfile(fname):
            self.get_video_stats()
        self.frame_range = []


    def get_video_stats(self,
                        stream: int = 0,
                        entries: Optional[tuple] = None,
                        verbose: bool = False) -> dict:
        """file statistics
            returns full stats, stores subset to self.stats
            if video stream has rotation, rotates width and height in subset

        stream  (int[0]) subset of stats for video stream
        entries (list,tuple [None]) extra queries
        verbose (bool [False])
        """
        if not osp.isfile(self.file):
            print(f"{self.file} not a valid file")

        _cmd = f"ffprobe -v quiet -print_format json -show_format -show_streams {self.file}"
        with os.popen(_cmd) as _fi:
            stats = json.loads(_fi.read())

        if 'streams' not in stats:
            print(f"No streams found\n{stats}")
            return stats

        videos = [s for s in stats['streams'] if s['codec_type'] == 'video']
        audios = [s for s in stats['streams'] if s['codec_type'] == 'audio']

        if stream >= len(videos):
            stream = len(videos) - 1
            print(f"only {len(videos)} streams found, returning stream {stream}")

        # subset video stream
        _stats = videos[stream]

        self.stats = {'type': 'video', 'file': self.file}
        self.stats['sar'] = 1.
        if 'sample_aspect_ratio' in _stats and _stats['sample_aspect_ratio'] != '1:1':
            _sar = [float(a) for a in (_stats['sample_aspect_ratio'].split(':')[0])]
            self.stats['sar'] = _sar[0]/_sar[1]


        if 'width' in _stats and 'height'in _stats:
            self.stats['width'] = _stats['width']
            self.stats['height'] = _stats['height']

            _rotated = ('tags' in _stats and 'rotate' in _stats['tags'] and
                        abs(eval(_stats['tags']['rotate'])) == 90)
            if _rotated:
                self.stats['width'] = _stats['height']
                self.stats['height'] = _stats['width']
        else:
            warnings.warn("'width' or 'height' not found in stats setting to 1080p or fix code") 
            self.stats['width'] = 1920
            self.stats['height'] = 1080

        _frame_rate_keys = [k for k in _stats if 'frame_rate' in k]
        if _frame_rate_keys:
            self.stats['rate'] = eval(_stats[_frame_rate_keys[0]])
        else:
            self.stats['rate'] = 30.0
            warnings.warn('no frame rate stats were found for stream, defaulting to 30 or fix code')

        if 'nb_frames' in _stats:
            self.stats['nb_frames'] = eval(_stats['nb_frames'])
        elif 'tags' in _stats and 'DURATION' in _stats['tags'] and 'rate' in self.stats:
            duration = _stats['tags']['DURATION']
            seconds = sum(x * float(t) for x, t in zip([3600, 60, 1], duration.split(":")))
            self.stats['nb_frames'] = round(seconds * self.stats['rate'])
        else:
            warnings.warn("'nb_frames' not found in stats, set manually or fix code") 
        _pad = 6
        if 'nb_frames' in self.stats:
            _pad = int(np.ceil(np.log10(self.stats['nb_frames'])))
        self.stats['pad'] = f"%0{_pad}d"

        if 'pix_fmt' in _stats:
            self.stats['pix_fmt'] = _stats['pix_fmt']

        if entries is not None:
            entries = [key for key in entries if key in _stats and key not in self.stats]
            for key in entries:
                self.stats[key] = eval(_stats[key])

        if verbose:
            print(_cmd)
            print(f"{len(audios)} audio streams, {len(videos)} video streams")
            print(f"\nStats for video stream: {stream}")
            print(json.dumps(self.stats, indent=2))
            print("\nFull stats")
            print(json.dumps(stats, indent=2))
        return stats


    def make_subtitles(self, name: Optional[str] = None, start_frame: int = 0) -> str:
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
        name = name if isinstance(name, str) else osp.splitext(osp.abspath(self.file))[0]
        name = name+'.srt' if '.srt' not in name else name
        sub = ""
        last_frame = self.frame_to_strftime(0)
        for i in range(self.stats['nb_frames']):
            _frame = i + start_frame
            next_frame = self.frame_to_strftime(_frame+1)
            sub += f"{_frame}\n{last_frame} --> {next_frame}\n\t{_frame}\t{last_frame}\n\n"
            last_frame = next_frame

        with open(name, 'w', encoding='utf8') as _fi:
            _fi.write(sub)
        return name


    def frame_to_time(self, frame: int = 0) -> float:
        """convert frame to time seconds
        Args
            frame   int
        """
        if not self.stats:
            self.get_video_stats(stream=0)
        return frame/self.stats['rate']

    def frame_to_strftime(self, frame: int = 0) -> str:
        """convert frame to time str HH:MM:SS.mmm
        Args
            frame   int
        """
        return self.strftime(self.frame_to_time(frame))

    @staticmethod
    def time_to_frame(intime: float, fps: float) -> int:
        """ return frame number for time in seconds
        Args
            intime  float
        """
        return int(round(intime * fps))

    @staticmethod
    def strftime(intime: float) -> str:
        """ time in seconds to HH:MM:SS.mmm
        Args
            intime  float
        """
        return '%02d:%02d:%02d.%03d'%((intime//3600)%24, (intime//60)%60, intime%60,
                                      (int((intime - int(intime))*1000)))
    @staticmethod
    def strftime_to_time(instftime: str) -> float:
        """ HH:MM:SS.mmm - > seconds
        """
        return round(sum(x * float(t) for x, t in zip([3600, 60, 1], instftime.split(":"))), 3)


    def build_filter_graph(self,
                           scale: Union[float, tuple] = 1.,
                           rate: Optional[float] = None,
                           crop: Optional[tuple[int,int,int,int]] = None,
                           subtitles: Optional[str] = None,
                           showframe: bool = False,
                           start_frame: int = 0,
                           scale_aspect_ratio: int = 0,
                           **kwargs) -> list:
        """ add filter graph to command -vf comma,sepparated,filters
        Args
            scale       (float, tuple [1])          -> scale=img_width*scale[0]:img_height*scale[1]
            rate        (float [None])              -> fps=rate
            crop        (tuple (w,h,x,y) [None])    -> crop=crop[0]:crop[1]}crop[2]:crop[3]
            subtitles   (str) .srt subtitles file   -> subtitles=subtitles
            showframe   (bool [False])              -> drawtext=text'%{n}'"
            start_frame (int [0]) for 'showframe arg'
                show_frame arg does not keep frames if when fast forwarding or rewinding
            scale_aspect_ratio  (int [0])   scale aspect ratio to 1:1
                if 1 scale[0] *= sar[0]/sar[1], if -1 scale[1] *= sar[1]/sar[0],

        """
        out = []
        _vf = []

        if not isinstance(scale, (list, tuple)):
            scale = (scale, scale)
        scale = list(scale)

        if scale_aspect_ratio != 0 and self.stats['sar'] != 1.:
            if scale_aspect_ratio > 0:
                scale[0] *= self.stats['sar']
            else:
                scale[1] /= self.stats['sar']

        if scale !=  [1.,1.]:
            self.stats['out_width'] = int(round(scale[0] * self.stats['width']))
            self.stats['out_height'] = int(round(scale[1] * self.stats['height']))
            _vf += [f"scale={self.stats['out_width']}:{self.stats['out_height']}"]

        if rate is not None:
            _vf += [f"fps={rate}"]

        if showframe:
            _snum="" if start_frame == 0 else f"start_number={start_frame}:"
            _vf += ["drawtext=x=w*0.01:y=h-2*th:fontcolor=white:fontsize=h*0.0185:"+
                    _snum+"text='%{n}'"]
        if subtitles is not None:
            if not osp.isfile(subtitles):
                _sub = subtitles
                subtitles = osp.join(osp.abspath(self.file).dirname, subtitles)
            assert osp.isfile(subtitles), f"subtitle_file not found <{_sub}>"
            _vf += [f'subtitles={subtitles}']
        if crop is not None:
            assert isinstance(crop, (list, tuple)) and len(crop) == 4, f"crop:(w,h,x,y) got {crop}"
            _vf += [f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}"]

        if _vf:
            out += ["-vf", ",".join(_vf)]
        return out


    def vlc(self,
            start: Union[int, float, str] = 0,
            nb_frames: Optional[int] = None,
            end:  Union[int, float, str, None] = None,
            **kwargs)-> None:
        """ ffplay does not do exact time search from playback use vlc instead
        """
        if not self.stats:
            self.get_video_stats(stream=kwargs.get('stream', 0))
        _io = self._start_end_time(start, nb_frames, end)
        if _io:
            _io = [f'--start-time={_io[0]}', f"--stop-time={_io[0]+_io[1]}"]
        _fcmd = ['cvlc'] + _io + [self.file]
        print(" ".join(_fcmd))
        # sp.call(_fcmd)
        proc = sp.Popen(_fcmd, stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE)
        proc.wait()
        proc.stdout.close()

        proc.terminate()


    def play(self,
             loop: int = 0,
             autoexit: bool = True,
             fullscreen: bool = False,
             noborder: bool = True,
             start: Union[int, float, str] = 0,
             nb_frames: Optional[int] = None,
             end: Union[int, float, str, None] =None,
             **kwargs)-> None:
        """ff play video

        ffplay -i metro.mov
        ffplay -start_number 5486 -i metro%08d.png
        Args
            loop        (int[0]) : number of loops, 0: forever
            autoexit    (bool [True]) close window on finish
            fullscreen  (bool [False])
            noborder    (bool[True])
            start       (int, float [0])
            nb_frames   (int [None])

        ** kwargs filter graph
            scale: float = 1.,
            step: int = 1, 
            rate: Optional[float] = None, 
            crop: Optional[tuple[int,int,int,int]] = None,
            subtitles: Optional[str] = None,
            showframe: bool = False,


        Does not accept exct values
        cvlc --start-time=<seconds> --stop_time=<seconds> <fname> works better
        """
        if start != 0:
            warnings.warn('.play()  cannont do exact time search, use .vlc() instead')
        if not self.stats:
            self.get_video_stats(stream=kwargs.get('stream', 0))

        _fcmd = [self.ffplay, "-loop", str(loop)]
        if autoexit:
            _fcmd += ["-autoexit"]
        if noborder:
            _fcmd += ["-noborder"]
        if fullscreen:
            _fcmd += ["-fs"]
        _fcmd += ['-i', self.file]
        _fcmd += self.format_start_end(start, nb_frames, end)

        if 'subtitles' in kwargs and isinstance(kwargs['subtitles'], bool) and kwargs['subtitles']:
            _dirname, _fname = osp.split(osp.abspath(self.file))
            _fname, _ext = osp.splitext(_fname)
            subtitles = [f.path for f in os.scandir(_dirname)
                         if f.name.endswith(".srt") and f.name[:len(_fname)] == _fname]
            if subtitles:
                kwargs['subtitles'] = subtitles[0]
            else:
                del kwargs['subtitles']

        _fcmd += self.build_filter_graph(start_frame=self.frame_range[0], **kwargs)


        print(" ".join(_fcmd))
        print("-------Interaction--------")
        print(" 'q', ESC        Quit")
        print(" 'f', LMDC       Full Screen")
        print(" 'p', SPACE      Pause")
        print(" '9'/'0'         Change Volume")
        print(" 's'             Step one frame")
        print("  RT,DN / LT,UP  10sec jump")
        print("  RMC            jump to percentage of film")
        print("--------------------------")

        # sp.call(_fcmd)
        #sp.Popen(_fcmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        proc = sp.Popen(_fcmd, stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE)
        proc.wait()
        proc.stdout.close()

    def playfiles(self, fname=None, folder=".", max_frames=None, fmt=('.jpg', '.jpeg', '.png'),):

        imgs, start = self._io.get_images(folder=folder, name=fname, fmt=fmt, max_imgs=max_frames)

        print(self.ffplay)
        print(start)
        print(self.ffplay)
        print(self.ffplay)
        if not imgs:
            return None

        _fcmd = [self.ffplay]
        if start or start is not None:
            _fcmd += ["-start_number", str(start)]
        _fcmd += ['-i', imgs]

        print(" ".join(_fcmd))
        print("-------Interaction--------")
        print(" 'q', ESC        Quit")
        print(" 'f', LMDC       Full Screen")
        print(" 'p', SPACE      Pause")
        print(" '9'/'0'         Change Volume")
        print(" 's'             Step one frame")
        print("  RT,DN / LT,UP  10sec jump")
        print("  RMC            jump to percentage of film")
        print("--------------------------")
        sp.call(_fcmd)

        return True

    def get_size(self, size: Union[str, tuple, int, None] = None) -> str:
        """ converts size to str 
        """
        if size is not None:
            if isinstance(size, str) and size.isnumeric(size):
                size = int(size)
            if isinstance(size, int):
                size = (size, size)
            if isinstance(size, (tuple, list)):
                size = f"{size[0]}x{size[1]}"
            if not isinstance(size, str) and (size.split('x')) == 2:
                raise NotImplementedError(f"size={size} invalid, \
                                          accepts: (int, int), int, 'intxint'")
        return size


    def stitch(self, dst, src, audio, fps, size, start_img, max_img, pix_fmt="yuv420p"):
        """
        HAS NOT BEEN REVISED
        ffmpeg -r 29.97 -start_number 5468 -i metro%08d.png -vcodec libx264 -pix_fmt yuv420p \
              /home/z/metropolis_color.mov
        ffmpeg -r 29.97 -start_number 5468 -i metro%08d.png -vcodec libx264 -pix_fmt yuv420p \
            -vframes 200 /home/z/metro_col.mov #only 200 frames
        """
        _ff = 'ffmpeg' if platform.system() != "Windows" else 'ffmpeg.exe'
        _fcmd = [_ff, '-r', str(fps)]

        # has to be before input
        if start_img is not None:
            _fcmd += ['-start_number', str(start_img)]

        _fcmd += ['-i', src]

        if audio is not None:
            _fcmd += ['-i', audio]

        size = self.get_size(size)
        if size is not None:
            _fcmd += ['-s', size]

        # codecs
        _fcmd += ['-vcodec', 'libx264']
        if audio is not None:
            _fcmd += ['-acodec', 'copy']

        _fcmd += ['-pix_fmt', pix_fmt]
        _fcmd += [dst]

        # number of frames # has to be just before outpyut
        if max_img is not None:
            _fcmd += ['-vframes', str(max_img)]

        print(" ".join(_fcmd))
        sp.call(_fcmd)


    def _export(self,
                start: Union[int, float, str] = 0,
                nb_frames: Optional[int] = None,
                stream: int = 0,
                end: Union[int, float, str, None] = None,
                **kwargs) -> str:
        """ returns export command stub and start_frame
        Args
            start       (int, float, str) # float is interpreted as time, int as frame
            nb_frames   (int)   number of frames from start
            end         (int, float, str) overrides nb_frames 
        """
        if not self.stats:
            self.get_video_stats(stream=stream)

        _interval = self.format_start_end(start, nb_frames, end)

        cmd =  [self.ffmpeg, '-i', self.file] + _interval
        cmd += self.build_filter_graph(start_frame=self.frame_range[0], **kwargs)
        return cmd


    def anytime_to_frame_time(self,
                              in_time: Union[int, float, str],
                              fps: Optional[float] = None) -> tuple[int, float]:
        """ float seconds, HH:MM:SS.mmm str or int frame to (int frame, float seconds)
        Args
            in_time     (str 'HH:MM:SS.mmm', float, int)
            fps         (float)
        """
        fps = fps if fps is not None else self.stats['rate']
        if isinstance(in_time, str):
            in_time = self.strftime_to_time(in_time)

        if isinstance(in_time, float):
            out_time = in_time
            out_frame = self.time_to_frame(in_time, fps)
        else: # int
            out_frame = in_time
            out_time = self.frame_to_time(in_time)

        return out_frame, out_time


    def _start_end_time(self,
                        start: Union[int, float, str] = 0,
                        nb_frames: Optional[int] = None,
                        end: Union[int, float, str, None] = None,
                        fps: Optional[float] = None) -> list:
        """
        Args
            start       (int, float, str) # float is interpreted as time, int as frame
            nb_frames   (int)   number of frames from start
            end         (int, float, str) overrides nb_frames
        """
        fps = fps if fps is not None else self.stats['rate']

        out = []
        start_frame = 0
        if start or nb_frames is not None or end is not None:
            start_frame, start_time = self.anytime_to_frame_time(start, fps)

            _max = self.stats['nb_frames'] - start_frame
            if end is not None:
                end_frame, end_time = self.anytime_to_frame_time(end, fps)
                nb_frames = min(end_frame - start_frame, _max)
                assert nb_frames > 0, f"end time {end} must be later than start {start}"
            elif nb_frames is not None:
                nb_frames = min(_max, nb_frames)
            else:
                nb_frames = _max
            end_time = self.frame_to_time(nb_frames)
            out = [start_time, end_time]
        self.frame_range = [start_frame, nb_frames]
        return out


    def format_start_end(self,
                         start: Union[int, float, str] = 0,
                         nb_frames: Optional[int] = None,
                         end: Union[int, float, str, None] = None,
                         fps: Optional[float] = None) -> list:
        """ if start > 0 or nb_frames  != None: [--ss seconds -t seconds]

        Args
            start       (int, float, str) # float is interpreted as time, int as frame
            nb_frames   (int)   number of frames from start
            end         (int, float, str) overrides nb_frames
        """
        out = self._start_end_time(start, nb_frames, end, fps)
        if out:
            out = ['-ss', str(out[0]), '-t', str(out[1])]
        return out


    def export_frames(self,
                      out_name: Optional[str] = None,
                      start: Union[int, float] = 0,
                      nb_frames: Optional[int] = None,
                      end: Union[int, float, str, None] = None,
                      scale: Union[float, tuple[float, float]] = 1.,
                      stream: int = 0,
                      out_folder: str = None,
                      scale_aspect_ratio: int = -1,
                      **kwargs) -> str:
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
                default: [-1] scales extended dimension down, 1 scales shorter dimension up, 0 nothing

        kwargs
            out_format  (str) in (".png", ".jpg", ".bmp") format override
            crop        (list, tuple (w,h,x,y)
        """
        cmd = self._export(start=start, nb_frames=nb_frames, end=end, scale=scale,
                           scale_aspect_ratio=scale_aspect_ratio,
                           stream=stream, crop=kwargs.get('crop', None))

        start_frame, nb_frames = self.frame_range

        # resolve name
        out_name = out_name if out_name is not None else self.file
        out_name, out_format = osp.splitext(out_name)
        out_format = kwargs.get('out_format', out_format)
        if out_format.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            out_format = ".png"

        if scale not in (1, (1, 1)):
            out_name += f"_{'-'.join(scale)}" if isinstance(scale, (list,tuple)) else f"_{scale}"

        out_name = f"{out_name}_{self.stats['pad']}{out_format}"

        if out_folder is not None:
            out_folder = osp.abspath(out_folder)
            os.makedirs(out_folder, exist_ok=True)
            out_name = osp.join(out_folder, osp.basename(out_name))
        cmd.append(out_name)

        _msg = ["exporting", out_name%(start_frame)]
        if nb_frames > 1:
            _msg += ["to ", out_name%(start_frame + nb_frames)]
        print(*_msg, " ...")

        proc = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE)
        proc.wait()

        out_files = [] # rename files, from  000..1, 000..N to -> 000..start_frame, .
        if start_frame != 0:
            for i in range(nb_frames-1, -1, -1):
                os.rename(out_name%(i+1), out_name%(i+start_frame))
                out_files.append(out_name%(i+start_frame))

        proc.stdout.close()

        return cmd, out_files


    def export_clip(self,
                    out_name: Optional[str] = None,
                    start: Union[int, float, str] = 0,
                    nb_frames: Optional[int] = None,
                    end: Union[int, float, str, None] = None,
                    scale: Union[float, tuple[float, float]] = 1.,
                    step: int = 1,
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
        TODO: generate intermediate video with keyframes then cut
        ffmpeg -i a.mp4 -force_key_frames 00:00:09,00:00:12 out.mp4

        """
        cmd = self._export(start=start, nb_frames=nb_frames, end=end, scale=scale,
                           step=step, stream=stream, crop=kwargs.get('crop', None))


        _name, _format = osp.splitext(self.file)
        out_format = None
        if out_name is None:
            start_frame, nb_frames = self.frame_range
            out_name = f"{_name}_{start_frame}-{nb_frames+start_frame}"
        else:
            out_name, out_format = osp.splitext(out_name)

        out_format = kwargs.get('out_format', _format if out_format is None else out_format)

        if scale not in (1, (1, 1)):
            out_name += f"_{'-'.join(scale)}" if isinstance(scale, (list,tuple)) else f"_{scale}"

        out_name = f"{out_name}{out_format}"

        if out_folder is not None:
            out_folder = osp.abspath(out_folder)
            os.makedirs(out_folder, exist_ok=True)
            out_name = osp.join(out_folder, osp.basename(out_name))


        if osp.isfile(out_name):
            assert overwrite, "file exists set overwrite=True"
            cmd.insert(1, '-y')

        if out_format == '.gif' or out_format == '.webm':
            cmd = cmd + ['-bitrate', '3000k']
        else:
            cmd += ["-c:a", "copy"]

        cmd.append(out_name)
        print(cmd)
        sp.call(cmd)
        # proc = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE)
        # proc.wait()

        return osp.abspath(out_name)


    def fits_in_memory(self, nb_frames=None, dtype_size=1, with_grad=0, scale=1, stream=0,
                       memory_type="GPU", step=1) -> int:
        """
            returns max number of frames that fit in memory
        """
        if not self.stats:
            self.get_video_stats(stream=stream)

        if nb_frames is None:
            nb_frames = self.stats['nb_frames']
        nb_frames = min(self.stats['nb_frames'], nb_frames)

        _w = self.stats['width'] * scale
        _h = self.stats['height'] * scale
        _requested_ram = (nb_frames * _w * _h * dtype_size * (1 + with_grad))/step//2**20
        _available_ram = CPUse().available if memory_type == "CPU" else GPUse().available

        if _requested_ram > _available_ram:
            max_frames = int(nb_frames * _available_ram // _requested_ram)
            _msg = f"{Col.B}Cannot load [{nb_frames},{_h},{_w},3] frames in {memory_type} {Col.RB}"
            _msg += f"{_available_ram} MB{Col.YB}, loading only {max_frames} frames {Col.AU}"
            print(_msg)
            nb_frames = max_frames
        return nb_frames


    def view_frame(self,
                   start: Union[int, float],
                   stream: int = 0,
                   **kwargs) -> None:
        """ open stream export frame to numpy, display with matplotlib

        Args:
            start       (int|float [0])  float: time in seconds, int: frame number
        """
        frame = self.to_numpy(start, nb_frames=1, stream=stream)

        plt.figure(figsize=kwargs.get('figsize', (15, 15*self.stats['height']/self.stats['width'])))
        if 'title' in kwargs:
            plt.title(kwargs['title'])
        plt.imshow(frame[0])
        if 'axis' in kwargs:
            plt.axis(kwargs['axis'])
        plt.show()


    def to_numpy(self,
                 start: Union[int, float, str] = 0,
                 nb_frames: Optional[int] = None,
                 end: Union[int, float, str, None] = None,
                 scale: Union[float, tuple] = 1.,
                 stream: int = 0,
                 step: int = 1,
                 dtype: np.dtype = np.uint8,
                 scale_aspect_ratio: int = -1,
                 **kwargs) -> np.ndarray:
        """
        read video to numpy
        Args
            start       (int|float [0])  float: seconds, int: frame, str: HH:MM:SS.mmm
            nb_frames   (int [None]) None: all frames
            end         (int, float, str, None) overrides nb_frames
            scale       (float [1])
            stream      (int [0]) video stream to load
            step        (int [1]) step thru video

        TODO check input depth bytes, will fail if not 24bpp
        """
        if not self.stats:
            self.get_video_stats(stream=stream)

        _fcmd = [self.ffmpeg, '-i', self.file]
        _fcmd += self.format_start_end(start, nb_frames, end)
        _fcmd += self.build_filter_graph(start_frame=self.frame_range[0], scale=scale,
                                         scale_aspect_ratio=scale_aspect_ratio, **kwargs)
        _fcmd += ['-start_number', str(self.frame_range[0]), '-f', 'rawvideo', '-pix_fmt', 'rgb24']
        _fcmd += ['pipe:']

        if 'crop' in kwargs:
            width = kwargs['crop'][0]
            height = kwargs['crop'][1]
        else:
            width = self.stats.get('out_width', self.stats['width'])
            height = self.stats.get('out_hight', self.stats['height'])

        bufsize = width*height*3

        nb_frames = self.fits_in_memory(self.frame_range[1], dtype_size=np.dtype(dtype).itemsize,
                                        scale=scale, stream=stream, memory_type='CPU', step=step)

        to_frame = min(self.stats['nb_frames'], nb_frames + self.frame_range[0])

        proc = sp.Popen(_fcmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=bufsize)
        out = self._to_numpy_proc(self.frame_range[0], to_frame, step, width, height, dtype,
                                  bufsize, proc)
        proc.wait()
        proc.stdout.close()

        return out


    def _to_numpy_proc(self,
                       start: int,
                       to_frame: int,
                       step: int,
                       width: int,
                       height: int,
                       dtype: np.dtype,
                       bufsize: int,
                       proc: sp.Popen) -> np.ndarray:
        """ read nb_frames at step from open pipe return ndarray [N,H,W,C] 
        """
        out = []
        for i in range(start, to_frame):
            if not i%step:
                buffer = proc.stdout.read(bufsize)
                if len(buffer) != bufsize:
                    break
                frame = np.frombuffer(buffer, dtype=np.uint8)
                out += [frame.reshape(height, width, 3).astype(dtype)]

        out = out[0][None] if len(out) == 1 else np.stack(out, axis=0)

        # TODO: this assumes that video input is uint8n: validate pixel format
        if dtype in (np.float32, np.float64):
            out /= 255.
        del buffer

        return out





## NOTES
        # def stream(stream_spec, cmd='ffmpeg', capture_stderr=False, input=None, quiet=False, overwrite_output=False):
#     args = compile(stream_spec, cmd, overwrite_output=overwrite_output)

#     # calculate framezie
#     framesize = _get_frame_size(stream_spec)

#     stdin_stream = subprocess.PIPE if input else None
#     stdout_stream = subprocess.PIPE
#     stderr_stream = subprocess.PIPE if capture_stderr or quiet else None
#     p = subprocess.Popen(args, stdin=stdin_stream, stdout=stdout_stream, stderr=stderr_stream)

#     while p.poll() is None:
#         yield _read_frame(p, framesize)

# def _read_frame(process, framesize):
#     return process.stdout.read(framesize)


# lossless
# ffmpeg -i left.avi -i right.avi -filter_complex hstack -c:v ffv1 output.avi
# # lossy
# ffmpeg -i left.avi -i right.avi -filter_complex "hstack,format=yuv420p" -c:v libx264 -crf 18 output.mp4
# # audio
# # ffmpeg -i left.avi -i right.avi -filter_complex "[0:v][1:v]hstack,format=yuv420p[v];[0:a][1:a]amerge[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 18 -ac 2 output.mp4

# snippets
# https://trac.ffmpeg.org/wiki/FFprobeTips

# # extract 3 frames at second 7
# ffmpeg -i MUCBCN.mp4 -ss 00:00:07.000 -vframes 3 thumb%04d.jpg -hide_banner #70KB
# ffmpeg -i MUCBCN.mp4 -ss 00:00:07.000 -vframes 3 thumb%04d.png -hide_banner #2MB
# ffmpeg -i MUCBCN.mp4 -ss 00:00:07.000 -vframes 3 thumb%04d.bmp -hide_banner #6MB

# # extract and ensure renumbering is frame 
# ffmpeg -i 'MUCBCN.mp4' -ss 00:00:12.000 -start_number 299 -vframes 3 thumb%04d.jpg -hide_banner

# # extract as video 
# ffmpeg -ss 120.2 -t 0.75 -i MUCBCN.mp4 out_muc.mp4 #use h264 default
# # ffmpeg -ss 120.2 -t 1.75 -i MUCBCN.mp4 out_muc.mp4 #-vcodec copy or -c:v copy fail to set the right timeline

# #extract as webm
# # No bit rate set. Defaulting to 96000 bps. - good bit rate 2000 to 3000 ; resize video - 

# # scaling
# ffmpeg -i input.jpg -vf scale=320:-1 output_320.png # keep aspect ration

# ffmpeg -i input.jpg -vf "scale=iw/2:ih/2" input_half_size.png
# ffmpeg -i input.jpg -vf "scale='min(320,iw)':'min(240,ih)'" input_not_upscaled.png
# # scale, clip time
# ffmpeg -ss 120.0 -t 3.0 -i MUCBCN.mp4 -vf "scale='min(320,iw/4)':-1" output.webm

# #ok
# sp.Popen(['ffmpeg', '-i', 'MUCBCN.mp4', '-ss', '00:00:20.000', '-t', '12.0', 'MUCBCN_500-800.mp4'], stdin=sp.PIPE, stderr=sp.PIPE)
# #fail
# sp.Popen(['ffmpeg', '-i', 'MUCBCN.mp4', '-ss', '00:00:20.000', '-t', '11.0', '-vf', 'scale=iw/.25:-1', 'MUCBCN_500-800_.25.mp4'], stdin=sp.PIPE, stderr=sp.PIPE)
# #ok
# ffmpeg -ss 00:00:20.000 -t 12.0 -i MUCBCN.mp4 -vf "scale=iw/2:-1" output.mp4
# ffmpeg -ss 00:00:20.000 -t 12.0 -i MUCBCN.mp4 -vf scale=iw/2:-1 output.mp4

# #play
# #half speed (audio doen not chnage speed)
# ffplay MUCBCN.mp4 -vf "setpts=2*PTS"


# # gif export
# ffmpeg -i INPUT -loop 10 -final_delay 500 out.gif
#     loop -1 none, 0 forever

# #https://ffmpeg.org/ffmpeg-formats.html

# #optimize gif export: generate palette / doesnt really work...
# ffmpeg -y -i INPUT -vf palettegen palette.png
# ffmpeg -y -i INPUT -i palette.png -filter_complex paletteuse OUT.gif
