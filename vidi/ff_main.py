"""File to handle all ffmpeg requests"""
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

    """
    def __init__(self, fname=None):

        self.ffplay = 'ffplay'
        self.ffmpeg = 'ffmpeg'
        self.ffprobe = 'ffprobe'
        self.stats = {}
        self._io = IO()
        self.file = fname
        if fname is not None and osp.isfile(fname):
            self.get_video_stats()

    def get_video_stats(self, stream=0, entries=None, verbose=False):
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

    def make_subtitle(self):
        """ 
        """
        name = f"{osp.splitext(osp.abspath(self.file))[0]}_frames.srt"
        sub = ""
        last_frame = self.frame_to_time(0)
        for i in range(self.stats['nb_frames']):
            next_frame = self.frame_to_time(i+1)
            sub += f"{i+1}\n{last_frame} --> {next_frame}\n\t{i+1}\t{last_frame}\n\n"
            last_frame = next_frame

        with open(name, 'w', encoding='utf8') as _fi:
            _fi.write(sub)


    def frame_to_time(self, frame=0):
        """convert frame number to time"""
        if not self.stats:
            self.get_video_stats(stream=0)
        outtime = frame/self.stats['rate']
        return self.strftime(outtime)

    def time_to_frame(self, intime):
        frame = int(intime * self.stats['rate'])
        return frame

    def strftime(self, intime):
        return '%02d:%02d:%02d.%03d'%((intime//3600)%24, (intime//60)%60, intime%60, (int((intime - int(intime))*1000)))

    def play(self, loop=0, autoexit=True, fullscreen=False, noborder=True, showframe=False, fontcolor="white"):
        """ff play video

        ffplay -i metro.mov
        ffplay -start_number 5486 -i metro%08d.png
        Args
            loop        (int[0]) : number of loops, 0: forever
            autoexit    (bool [True]) close window on finish
            fullscreen  (bool [False])
            noborder    (bool[True])
            showframe   (bool[False]): draw current frame number
            fontcolor   (str [white])

        """
        if not self.stats:
            self.get_video_stats(stream=0)

        _fcmd = [self.ffplay, "-loop", str(loop)]
        if autoexit:
            _fcmd += ["-autoexit"]
        if noborder:
            _fcmd += ["-noborder"]
        if fullscreen:
            _fcmd += ["-fs"]
        _fcmd += ['-i', self.file ]
        if showframe:
            _cmd = f"drawtext=fontfile=Arial.ttf: x=(w-tw)*0.98: y=h-(2*lh): fontcolor={fontcolor}: fontsize=h*0.0185: " + "text='%{n}'"
            _fcmd += ["-vf", _cmd]   

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
        #sp.Popen(_fcmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)

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

    def get_size(self, size=None):
        if size is not None:
            if isinstance(size, str):
                if 'x' not in size:
                    assert size.isnumeric(size), "size can be tuple of ints, string of ints sepparated by 'x' or single int"
                    size = int(size)
            if isinstance(size, int):
                size = (size, size)
            if isinstance(size, (tuple, list)):
                size = "%dx%d"%size
        return size

    def stitch(self, dst, src, audio, fps, size, start_img, max_img, pix_fmt="yuv420p"):
        """
        ffmpeg -r 29.97 -start_number 5468 -i metro%08d.png -vcodec libx264 -pix_fmt yuv420p /home/z/metropolis_color.mov
        ffmpeg -r 29.97 -start_number 5468 -i metro%08d.png -vcodec libx264 -pix_fmt yuv420p -vframes 200 /home/z/metro_col.mov #only 200 frames
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
                start: Union[int, float] = 0,
                nb_frames: Optional[int] = None,
                scale: float = 1.,
                step: int = 1,
                stream: int = 0,
                crop: Union[list, tuple, None] = None) -> str:
        """ common to clip and frame exports
        Args
            start   (int, float) # float is interpreted as time, int as frame
            nb_frames   
        """
        if not self.stats:
            self.get_video_stats(stream=stream)
        _rate = self.stats['rate']
        _height = self.stats['height']
        _width = self.stats['width']

        # range start, frame and time
        if isinstance(start, float):
            time_start = self.strftime(start)
            start = self.time_to_frame(start)
        else:
            time_start = self.frame_to_time(start)

        # range size, number  and time
        _max_frames = self.stats['nb_frames'] - start
        if nb_frames is None:
            nb_frames = _max_frames
        else:
            nb_frames = min(_max_frames, nb_frames)
        time_end = str(nb_frames/_rate)

        # export command
        # '-vframes', str(nb_frames)]
        cmd =  [self.ffmpeg, '-i', self.file, '-ss', time_start, '-t', time_end]

        # resize
        if scale != 1:
            cmd += ['-s', f"{int(_width*scale)}x{int(_height*scale)}" ]

        if step > 1:
            cmd += ['-r', str(_rate // step), '-vf', 'mpdecimate,setpts=N/FRAME_RATE/TB']

        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            _crop = f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}"
            if step == 1:
                cmd += ["-vf", _crop]
            else:
                cmd[-1] += ", "+_crop

        return cmd

    def export_frames(self,
                      out_name: Optional[str] = None,
                      start: Union[int, float] = 0,
                      nb_frames: Optional[int] = None,
                      scale: float = 1.,
                      step: int = 1,
                      stream: int = 0,
                      out_folder=None,
                      **kwargs) -> str:
        """ extract frames from video
        Args
            out_name    (str)   # if no format in name ".png
            start       (int|float [0])  float: time in seconds, int: frame number
            nb_frames   (int [None]):   default: to end of clip
            scale       (float [1]) rescale output
            stream      (int [0]) if more than one stream in video
            out_folder  optional, save to folder

        kwargs
            out_format  (str) in (".png", ".jpg", ".bmp") format override
            crop        (list, tuple (w,h,x,y)
        """
        cmd = self._export(start=start, nb_frames=nb_frames, scale=scale, step=step,
                           stream=stream, crop=kwargs.get('crop', None))

        # resolve name
        if out_name is None:
            out_name = osp.splitext(self.file)[0]


        out_name, out_format = osp.splitext(out_name)

        if "out_format" in kwargs:
            out_format = kwargs["out_format"]

        if out_format.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            out_format = ".png"

        if scale != 1:
            _pad = ""
            if "%0" in out_name:
                out_name, _pad = out_name.split("%0")
                if out_name[-1] == "_":
                    out_name = out_name[:-1]
                _pad = "_%0" + _pad
            out_name += f"_{scale}" + _pad

        if not "%0" in out_name:
            out_name += "_" + self.stats['pad']

        out_name += out_format
        if out_folder is not None:
            out_folder = osp.abspath(out_folder)
            os.makedirs(out_folder, exist_ok=True)
            out_name = osp.join(out_folder, out_name)

        cmd.append(out_name)
        print(" ".join(cmd))
        sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE)

        with open(out_name+".txt", 'w', encoding="utf8") as _fi:
            _fi.write(" ".join(cmd) +"\n")


        return osp.abspath(out_name)

    def export_clip(self, out_name=None, start=0, nb_frames=None, scale=1, step=1, stream=0, out_folder=None, **kwargs):
        """ extract video clip
        Args:
            out_name    (str [None]) default is <input_name>_framein_frame_out<input_format>
            start       (int | float [0])  default: start of input clip, if float: time
            nb_frames   (int [None]):      default: to end of clip
            scale       (float [1]) rescale output
            stream      (int [0]) if more than one stream in video
        kwargs
            out_format  (str) in (".mov", ".mp4") format override
            crop        (list, tuple (w,h,x,y)
        TODO: generate intermediate video with keyframes then cut
        ffmpeg -i a.mp4 -force_key_frames 00:00:09,00:00:12 out.mp4

        """
        cmd = self._export(start=start, nb_frames=nb_frames, scale=scale, step=step,
                           stream=stream, crop=kwargs.get('crop', None))

        # resolve name
        if out_name is None:
            if nb_frames is None:
                nb_frames = self.stats["nb_frames"]
            out_name = osp.splitext(self.file)[0]+ '_' + str(start) + '-' + str(nb_frames+start)

        # format
        out_name, out_format = osp.splitext(out_name)
        if not out_format:
            out_format = osp.splitext(self.file)[1]
        if "out_format" in kwargs:
            out_format = kwargs["out_format"]

        if scale != 1:
            out_name += f"_{scale}"

        out_name += out_format
        if out_folder is not None:
            out_folder = osp.abspath(out_folder)
            os.makedirs(out_folder, exist_ok=True)
            out_name = osp.join(out_folder, out_name)

        if out_format == '.gif' or out_format == '.webm':
            cmd = cmd + ['-bitrate', '3000k']

        cmd.append(out_name)
        print(" ".join(cmd))
        sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE)

        return osp.abspath(out_name)


    def fits_in_memory(self, nb_frames=None, dtype_size=1, with_grad=0, scale=1, stream=0,
                       memory_type="GPU", step=1):
        """
            returns max number of frames that fit in memory
        """
        if not self.stats:
            self.get_video_stats(stream=stream)

        nb_frames = self.stats['nb_frames'] if nb_frames is None else min(self.stats['nb_frames'],
                                                                          nb_frames)

        width = self.stats['width'] * scale
        height = self.stats['height'] * scale
        _requested_ram = (nb_frames * width * height * dtype_size * (1 + with_grad))/step//2**20
        _available_ram = CPUse().available if memory_type == "CPU" else GPUse().available

        if _requested_ram > _available_ram:
            max_frames = int(nb_frames * _available_ram // _requested_ram)
            _msg = f"{Col.B}Cannot load [{nb_frames},{height},{width},3] frames in {memory_type} {Col.RB}"
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

    # TODO: flawed - fix
    # def view_frames(self,
    #                start: Union[int, float],
    #                nb_frames: int = 10,
    #                step: int = 1,
    #                stream: int = 0,
    #                scale: float = 0.5,
    #                **kwargs) -> None:
    #     """ plot grid of frames
    #     """
    #     frames = self.to_numpy(start, nb_frames=nb_frames, step=step, scale=scale, stream=stream)
    #     cols = kwargs.get('cols', min(5, nb_frames))
    #     rows = kwargs.get('rows', np.ceil(nb_frames/cols))

    #     width = rows*self.stats['height']*20/cols*self.stats['width']
    #     plt.figure(figsize=kwargs.get('figsize', (20, width)))
    #     if 'title' in kwargs:
    #         plt.title(kwargs['title'])
    #     for i in rows:
    #         for j in cols:
    #             plt.subplot(rows, cols, 1+j+cols*i)
    #             plt.imshow(frames[i])
    #             if 'axis' in kwargs:
    #                 plt.axis(kwargs['axis'])
    #     plt.tight_layout()
    #     plt.show()


    def to_numpy(self,
                 start: Union[int, float] = 0,
                 nb_frames: Optional[int] = None,
                 scale: float = 1.,
                 stream: int = 0,
                 step: int = 1,
                 dtype: np.dtype = np.uint8,
                 memory_type: str = "CPU") -> np.ndarray:
        """
        read video to numpy
        Args
            start       (int|float [0])  float: time in seconds, int: frame number
            nb_frames   (int [None]) None: all frames
            scale       (float [1])
            stream      (int [0]) video stream to load
            step        (int [1]) step thru video

        TODO: loader iterator yield
        TODO: crop or transform
        TODO check input depth bytes, will fail if not 24bpp
        """
        if not self.stats:
            self.get_video_stats(stream=stream)

        if isinstance(start, float):
            _time = self.strftime(start)
            start = self.time_to_frame(start)
        else:
            _time = self.frame_to_time(start)


        _fcmd = [self.ffmpeg, '-i', self.file, '-ss', _time, '-start_number', str(start),
                 '-f', 'rawvideo', '-pix_fmt', 'rgb24']

        width = self.stats['width']
        height = self.stats['height']
        if scale != 1:
            width = int(self.stats['width'] * scale)
            height = int(self.stats['height'] * scale)
            _scale = ['-s', f'{width}x{height}']
            _fcmd = _fcmd + _scale
        _fcmd += ['pipe:']

        bufsize = width*height*3

        nb_frames = self.fits_in_memory(nb_frames, dtype_size=np.dtype(dtype).itemsize, scale=scale,
                                        stream=stream, memory_type=memory_type, step=step)

        to_frame = self.stats['nb_frames'] if nb_frames is None else min(self.stats['nb_frames'],
                                                                          nb_frames + start)

        proc = sp.Popen(_fcmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=bufsize)
        out = self._to_numpy_proc(start, to_frame, step, width, height, dtype, bufsize, proc)

        proc.stdout.close()
        proc.wait()

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

        # TODO: this assumes that video input is uint8 but thats not a given: validate pixel format
        if dtype in (np.float32, np.float64):
            out /= 255.
        del buffer

        return out

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


"""
# lossless
ffmpeg -i left.avi -i right.avi -filter_complex hstack -c:v ffv1 output.avi
# lossy
ffmpeg -i left.avi -i right.avi -filter_complex "hstack,format=yuv420p" -c:v libx264 -crf 18 output.mp4
# audio
# ffmpeg -i left.avi -i right.avi -filter_complex "[0:v][1:v]hstack,format=yuv420p[v];[0:a][1:a]amerge[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 18 -ac 2 output.mp4

snippets
https://trac.ffmpeg.org/wiki/FFprobeTips

# extract 3 frames at second 7
ffmpeg -i MUCBCN.mp4 -ss 00:00:07.000 -vframes 3 thumb%04d.jpg -hide_banner #70KB
ffmpeg -i MUCBCN.mp4 -ss 00:00:07.000 -vframes 3 thumb%04d.png -hide_banner #2MB
ffmpeg -i MUCBCN.mp4 -ss 00:00:07.000 -vframes 3 thumb%04d.bmp -hide_banner #6MB

# extract and ensure renumbering is frame 
ffmpeg -i 'MUCBCN.mp4' -ss 00:00:12.000 -start_number 299 -vframes 3 thumb%04d.jpg -hide_banner

# extract as video 
ffmpeg -ss 120.2 -t 0.75 -i MUCBCN.mp4 out_muc.mp4 #use h264 default
# ffmpeg -ss 120.2 -t 1.75 -i MUCBCN.mp4 out_muc.mp4 #-vcodec copy or -c:v copy fail to set the right timeline

#extract as webm
# No bit rate set. Defaulting to 96000 bps. - good bit rate 2000 to 3000 ; resize video - 

# scaling
ffmpeg -i input.jpg -vf scale=320:-1 output_320.png # keep aspect ration

ffmpeg -i input.jpg -vf "scale=iw/2:ih/2" input_half_size.png
ffmpeg -i input.jpg -vf "scale='min(320,iw)':'min(240,ih)'" input_not_upscaled.png
# scale, clip time
ffmpeg -ss 120.0 -t 3.0 -i MUCBCN.mp4 -vf "scale='min(320,iw/4)':-1" output.webm

#ok
sp.Popen(['ffmpeg', '-i', 'MUCBCN.mp4', '-ss', '00:00:20.000', '-t', '12.0', 'MUCBCN_500-800.mp4'], stdin=sp.PIPE, stderr=sp.PIPE)
#fail
sp.Popen(['ffmpeg', '-i', 'MUCBCN.mp4', '-ss', '00:00:20.000', '-t', '11.0', '-vf', 'scale=iw/.25:-1', 'MUCBCN_500-800_.25.mp4'], stdin=sp.PIPE, stderr=sp.PIPE)
#ok
ffmpeg -ss 00:00:20.000 -t 12.0 -i MUCBCN.mp4 -vf "scale=iw/2:-1" output.mp4
ffmpeg -ss 00:00:20.000 -t 12.0 -i MUCBCN.mp4 -vf scale=iw/2:-1 output.mp4

#play
#half speed (audio doen not chnage speed)
ffplay MUCBCN.mp4 -vf "setpts=2*PTS"


# gif export
ffmpeg -i INPUT -loop 10 -final_delay 500 out.gif
    loop -1 none, 0 forever

#https://ffmpeg.org/ffmpeg-formats.html

#optimize gif export: generate palette / doesnt really work...
ffmpeg -y -i INPUT -vf palettegen palette.png
ffmpeg -y -i INPUT -i palette.png -filter_complex paletteuse OUT.gif
"""