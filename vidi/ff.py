"""@xvdp
ffmpeg wratpper class

    V = FF(videofile)
    V.export_clip()
    V.export_frames()
    V.to_numpy()        # -> ndarray
    V.make_subtitles()  # frame number and time subtitles
    V.vlc()             # play with vlc
    V.play()            # ffplay
    V.stats             # ffprobe -> dict

"""
from typing import Union, Optional
import warnings
import os
import os.path as osp
import subprocess as sp
import json
import numpy as np
import matplotlib.pyplot as plt


from .utils import frame_to_time, anytime_to_frame_time
from .functional import read_frame, get_formats, check_format, _make_subtitles, get_encoders

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=suppressed-message

class FF():
    """wrapper class to ffmpeg, ffprobe, ffplay

    """
    def __init__(self, fname: str) -> None:

        assert osp.isfile(fname), f"video not found {fname}"

        self.ffplay = 'ffplay'
        self.ffmpeg = 'ffmpeg'
        self.ffprobe = 'ffprobe'
        self.stats = {}
        self.file = fname
        self.get_video_stats()
        self.frame_range = []
        self.fmts = get_formats()


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
            _sar = [float(a) for a in (_stats['sample_aspect_ratio'].split(':'))]
            self.stats['sar'] = _sar[0]/_sar[1]

        if 'bits_per_raw_sample' in stats:
            self.stats['bpp'] = int(_stats['bits_per_raw_sample'])
        else:
            _bps, *_ = check_format('yuv420p')
            self.stats['bpp'] = int(_bps[0])

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

        if 'tags' in _stats and 'timecode' in _stats['tags']:
            self.stats['timecode'] = _stats['tags']['timecode']
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
        """ make frame/time subtitle file .srt
        Args
            name        (str [None]) name of subtitle file: default fname.replace(<.ext>, '.srt')
            start_frame (int [0])
        """
        fps = self.stats['rate']
        nb_frames = self.stats['nb_frames']
        name = name if isinstance(name, str) else osp.splitext(osp.abspath(self.file))[0]
        name = name+'.srt' if '.srt' not in name else name
        return _make_subtitles(name, nb_frames, fps, start_frame)


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

        self.stats['out_width'] = self.stats['width']
        self.stats['out_height'] = self.stats['height']
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
            start_frame, start_time = anytime_to_frame_time(start, fps)

            _max = self.stats['nb_frames'] - start_frame
            if end is not None:
                end_frame, end_time = anytime_to_frame_time(end, fps)
                nb_frames = min(end_frame - start_frame, _max)
                assert nb_frames > 0, f"end time {end} must be later than start {start}"
            elif nb_frames is not None:
                nb_frames = min(_max, nb_frames)
            else:
                nb_frames = _max
            end_time = frame_to_time(nb_frames, fps)
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
                      start: Union[int, float] = 0,
                      nb_frames: Optional[int] = None,
                      end: Union[int, float, str, None] = None,
                      scale: Union[float, tuple[float, float]] = 1.,
                      stream: int = 0,
                      out_name: Optional[str] = None,
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
                default: -1: downscales extended dim; 1: up scales shorter dim; 0: no scaling

        kwargs
            out_format  (str) in (".png", ".jpg", ".bmp") format override
            crop        (tuple (w,h,x,y) [None])    -> crop=crop[0]:crop[1]}crop[2]:crop[3]
        """
        out_name = out_name if out_name is not None else self.file
        out_name, out_format = osp.splitext(out_name)
        if out_format == ".yuv":
            warnings.warn("yuv format only exports only one frame with ffmpeg.")

        cmd = self._export(start=start, nb_frames=nb_frames, end=end, scale=scale,
                           scale_aspect_ratio=scale_aspect_ratio,
                           stream=stream, crop=kwargs.get('crop', None))

        start_frame, nb_frames = self.frame_range

        # resolve name
        out_name = f"{out_name}_{self.stats['out_width']}_{self.stats['out_height']}"

        out_format = kwargs.get('out_format', out_format)
        if out_format.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".yuv"):
            out_format = ".png"
        if out_format == '.yuv':
            if 'yuv' in self.stats['pix_fmt']:
                out_name = f"{out_name}_{self.stats['pix_fmt']}"

        out_name = f"{out_name}_{self.stats['pad']}{out_format}"

        if out_folder is not None:
            out_folder = osp.abspath(out_folder)
            os.makedirs(out_folder, exist_ok=True)
            out_name = osp.join(out_folder, osp.basename(out_name))
        cmd.append(out_name)

        if 'verbose' in kwargs and kwargs['verbose']:
            _msg = ["exporting", out_name%(start_frame)]
            if nb_frames > 1:
                _msg += ["to ", out_name%(start_frame + nb_frames)]
            print(*_msg, " ...")
            print(" ".join(cmd))

        proc = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE)
        proc.wait()

        out_files = [] # rename files, from  000..1, 000..N to -> 000..start_frame, .
        if start_frame != 0:
            for i in range(nb_frames-1, -1, -1):
                os.rename(out_name%(i+1), out_name%(i+start_frame))
                out_files.append(out_name%(i+start_frame))

        proc.stdout.close()

        return cmd, out_files

    @staticmethod
    def _codec_checks(fmt, vcodec=None, acodec=None) -> tuple:
        """
        """
        vcodecs, acodecs = get_encoders()
        if vcodec is None:
            if fmt == ".webm":
                _vcs = [v for v in ['libvpx', 'libvpx-vp9'] if v in vcodecs]
                assert _vcs, ".webm libvpx codecs not found in ffmpeg version"
                vcodec = _vcs[0]
            elif fmt == ".ogg":
                assert 'libtheora' in vcodecs, ".ogg libtheora not found in ffmpeg version"
                vcodec = 'libtheora'
            elif fmt == ".mp4":
                _vcs = [v for v in ['libx264', 'libopenh264'] if v in vcodecs]
                assert _vcs, ".mp4 libx264 codecs not found in ffmpeg version"
                vcodec = _vcs[0]
            elif fmt == ".gif":
                pass
            else:
                vcodec = "copy"
        else:
            assert vcodec in vcodecs, f"encoder {vcodec} not found, use {vcodecs}"

        if acodec is None:
            if fmt == ".webm":
                assert 'libvorbis'  in acodecs, ".webm libvorbis not found in ffmpeg version"
                acodec = 'libvorbis'
            elif fmt == ".ogg":
                assert 'vorbis' in acodecs, ".ogg libtheora not found in ffmpeg version"
                acodec = 'vorbis'
            elif fmt == ".gif":
                pass
            else:
                acodec = "copy"
        else:
            assert acodec in acodecs, f"encoder {acodec} not found, use {acodecs}"
        return vcodec, acodec



    def export_clip(self,
                    out_name: Optional[str] = None,
                    start: Union[int, float, str] = 0,
                    nb_frames: Optional[int] = None,
                    end: Union[int, float, str, None] = None,
                    scale: Union[float, tuple[float, float]] = 1.,
                    vcodec: Optional[str] = None,
                    acodec: Optional[str] = None,
                    step: int = 1,
                    stream: int = 0,
                    out_folder: str = None,
                    overwrite: bool = False,
                    **kwargs):
        """ extract video clip - by default copies codecs
        Args:
            out_name    (str [None]) default is <input_name>_framein_frame_out<input_format>
            start       (int | float [0])  default: start of input clip, if float: time
            nb_frames   (int [None]):      default: to end of clip
            end         (int, float, str, None) overrides nb_frames
            scale       (float [1]) rescale output
            vcodec      (str [None]) default to copy unless for web (webm, ogg, mp4)
                for web use:
                    -vcodec libx24 + mp4  -b:v 8192k -f segment -segment_time 4
                    -vcodec libtheora + .ogg -acodec vorbis
                    -vcodec libvpx -acodec libvorbis
            vcodec      (str [None]) default to copy unless for web (webm, ogg, mp4)
            stream      (int [0]) if more than one stream in video
        kwargs
            out_format  (str) in (".mov", ".mp4") format override
            crop        (tuple (w,h,x,y) [None])    -> crop=crop[0]:crop[1]}crop[2]:crop[3]
            bitrate     (str) eg. 8192k
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

        vcodec, acodec = self._codec_checks(out_format, vcodec, acodec)
        if vcodec is not None:
            cmd += ["-c:v", vcodec]
        if acodec is not None:
            cmd += ["-c:a", acodec]

        if 'bitrate' in kwargs:
            cmd +=  ['-bitrate', kwargs['bitrate']]
        elif out_format in (".gif", ".webm"):
            cmd +=  ['-bitrate', "3000K"]
        elif out_format == ".mp4":
            cmd +=  ['-bitrate', "8192K"]

        if osp.isfile(out_name):
            assert overwrite, "file exists set overwrite=True"
            cmd.insert(1, '-y')


        cmd.append(out_name)
        if nb_frames is None:
            nb_frames = self.stats['nb_frames'] - start

        print(f"exporting clip {out_name} frames ({start}-{nb_frames+start})\n {cmd}")
        sp.call(cmd)
        # proc = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE)
        # proc.wait()

        return osp.abspath(out_name)


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

    def _get_bufsize(self,
                     bits: tuple,
                     nb_frames: int = 1,
                     crop: Optional[tuple[int,int,int,int]] = None) -> int:
        if crop is not None:
            self.stats['out_width'] = crop[0]
            self.stats['out_height'] = crop[1]
        _fmt_byte_depth =  np.ceil(bits[0]/8)
        _fmt_planes = sum(bits)/bits[0]
        _area = self.stats['out_height'] * self.stats['out_width']
        return int(_area * nb_frames* _fmt_byte_depth * _fmt_planes)


    def to_numpy(self,
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
        if not self.stats:
            self.get_video_stats(stream=stream)
        if compressed and to_rgb:
            compressed = False
        assert start < self.stats['nb_frames'], f"start={start} is larger than \
            nb_frames {self.stats['nb_frames']}"

        pix_fmt = kwargs.get('pix_fmt', self.stats['pix_fmt'])
        assert pix_fmt in self.fmts, f"pix_fmt {pix_fmt} not in:  {sorted(self.fmts.keys())}"
        channel_axis = kwargs.get('channel_axis', -1)
        _transpose = False
        if to_rgb and 'yuv' in pix_fmt and channel_axis != -1:
            channel_axis = -1
            _transpose = True

        _fcmd = [self.ffmpeg, '-i', self.file]
        _fcmd += self.format_start_end(start, nb_frames, end)
        _fcmd += self.build_filter_graph(start_frame=self.frame_range[0], scale=scale,
                                         scale_aspect_ratio=scale_aspect_ratio, **kwargs)
        _fcmd += ['-start_number', str(self.frame_range[0]), '-f', 'rawvideo',
                  '-pix_fmt', pix_fmt, 'pipe:']

        nb_frames = self.frame_range[1]
        bits, *_ = check_format(pix_fmt)
        bufsize = self._get_bufsize(bits, nb_frames=nb_frames, crop=crop)

        # TODO read frame by frame instead of whole chunk
        proc = sp.Popen(_fcmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=bufsize)
        buffer = proc.stdout.read(bufsize)
        proc.communicate()
        proc.wait()
        proc.stdout.close()

        out = []

        for i in range(nb_frames):
            fro = i * bufsize//nb_frames
            _to = (i + 1) * bufsize//nb_frames
            frame = read_frame(buffer[fro:_to], pix_fmt, self.stats['out_width'],
                               self.stats['out_height'], channel_axis,
                               kwargs.get('interpolation', 1), compressed,
                               to_rgb, dtype)
            out.append(frame)
        del buffer

        # output stack
        if isinstance(out[0], np.ndarray):
            out = np.stack(out, axis=0)
            if _transpose:
                out = out.transpose(0,3,1,2).copy(order='C')

        return out


    #   TODO REVISE
    #,  from .utils import Col, CPUse, GPUse
    # def fits_in_memory(self, nb_frames=None, dtype_size=1, with_grad=0, scale=1, stream=0,
    #                    memory_type="GPU", step=1) -> int:
    #     """
    #         returns max number of frames that fit in memory
    #     """
    #     if not self.stats:
    #         self.get_video_stats(stream=stream)

    #     if nb_frames is None:
    #         nb_frames = self.stats['nb_frames']
    #     nb_frames = min(self.stats['nb_frames'], nb_frames)

    #     _w = self.stats['width'] * scale
    #     _h = self.stats['height'] * scale
    #     _requested_ram = (nb_frames * _w * _h * dtype_size * (1 + with_grad))/step//2**20
    #     _available_ram = CPUse().available if memory_type == "CPU" else GPUse().available

    #     if _requested_ram > _available_ram:
    #         max_frames = int(nb_frames * _available_ram // _requested_ram)
    #         _msg = f"{Col.B}Cannot load [{nb_frames},{_h},{_w},3] frames in {memory_type} {Col.RB}"
    #         _msg += f"{_available_ram} MB{Col.YB}, loading only {max_frames} frames {Col.AU}"
    #         print(_msg)
    #         nb_frames = max_frames
    #     return nb_frames

    # def playfiles(self, fname=None, folder=".", max_frames=None, fmt=('.jpg', '.jpeg', '.png'),):

    #     imgs, start = self._io.get_images(folder=folder, name=fname, fmt=fmt, max_imgs=max_frames)

    #     print(self.ffplay)
    #     print(start)
    #     print(self.ffplay)
    #     print(self.ffplay)
    #     if not imgs:
    #         return None

    #     _fcmd = [self.ffplay]
    #     if start or start is not None:
    #         _fcmd += ["-start_number", str(start)]
    #     _fcmd += ['-i', imgs]

    #     print(" ".join(_fcmd))
    #     print("-------Interaction--------")
    #     print(" 'q', ESC        Quit")
    #     print(" 'f', LMDC       Full Screen")
    #     print(" 'p', SPACE      Pause")
    #     print(" '9'/'0'         Change Volume")
    #     print(" 's'             Step one frame")
    #     print("  RT,DN / LT,UP  10sec jump")
    #     print("  RMC            jump to percentage of film")
    #     print("--------------------------")
    #     sp.call(_fcmd)

    #     return True