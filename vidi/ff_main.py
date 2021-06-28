"""File to handle all ffmpeg requests"""
import os
import os.path as osp
import subprocess as sp
from platform import system
import numpy as np
import json
from .io_main import IO

class FF():
    """wrapper class to ffmpeg, ffprobe, ffplay
    #TODO inherit VIDI and IO

        Examples:
            >>> from vidi import FF
            >>> f = FF('MUCBCN.mp4')
            >>> print(f.stats)
            >>> c = f.clip(start_frame=100, num_frames=200) #output video clip
            # output scalled reformated video clip
            >>> d = f.clip(start_frame=500, num_frames=300, out_format=".webm", scale=0.25)
            # output images
            >>> e = f.get_frames(out_format='.png', start_frame=100, num_frames=5, scale=0.6)
            >>> f.play(c)

    """
    def __init__(self, fname=None):

        self.ffplay = 'ffplay'
        self.ffmpeg = 'ffmpeg'
        self.ffprobe = 'ffprobe'
        self._if_win()
        self.stats = {}
        self._io = IO()
        self.file = fname
        #self._resolve_filen(fname)

    def help(self):
        """print examples"""
        _help = """
            >>> from vidi import FF
            >>> f = FF('MUCBCN.mp4')
            >>> print(f.stats)
            >>> c = f.export_clip(start_frame=100, num_frames=200) #output video clip
            # output scalled reformated video clip
            >>> d = f.export_clip(start_frame=500, num_frames=300, out_format=".webm", scale=0.25)
            # output images
            >>> e = f.export_frames(out_format='.png', start_frame=100, num_frames=5, scale=0.6)
            >>> f.play(c)
        """.format()
        print(_help)

    def _resolve_filen(self, fname=None):
        # only clobber self.file is fname exists
        if fname is None:
            fname = self.file
        self.file = self._io.file_resolve(fname)
        if self.file is not None:
            self._get_stats()

    def _valid(self, fname=None):
        self._resolve_filen(fname)
        assert self.file is not None, 'enter valid file'

    def _if_win(self):
        if system == 'Windows':
            self.ffplay = 'ffplay.exe'
            self.ffmpeg = 'ffmpeg.exe'
            self.ffprobe = 'ffprobe.exe'

    def get_stats(self):
        """prints file statistics"""
        self._get_stats()
        print(json.dumps(self.stats, indent=2))

    def _get_stats(self):
        """file statistics
            Fix, some file types are missing these stats, eg. webm
        """
        _ext = osp.splitext(self.file)
        if _ext in ['.gif', '.webm']:
            pass
            #TODO implement full read of file to get stats
        else:
            cmd = ' -v error -show_entries stream=width,height,avg_frame_rate,nb_frames -of default=noprint_wrappers=1:nokey=1 '
            _p = os.popen(self.ffprobe + cmd +self.file)

            try:
                self.stats['width'] = eval(_p.readline())
                self.stats['height'] = eval(_p.readline())
                try:
                    self.stats['rate'] = eval(_p.readline())
                    self.stats['frames'] = eval(_p.readline())
                    self.stats['pad'] = "%%0%dd" %len(str(self.stats['frames']))
                    self.stats['type'] = 'video'
                except:
                    self.stats['type'] = 'image'
                self.stats['file'] = self.file
            except:
                print(self.file, 'is not a valid file')


    def frame_to_time(self, frame=0):
        """convert frame number to time"""
        self._valid()
        outtime = frame/self.stats['rate']
        return self.strftime(outtime)

    def time_to_frame(self, intime):
        frame = int(intime * self.stats['rate'])
        return frame

    def strftime(self, intime):
        return '%02d:%02d:%02d.%03d'%((intime//3600)%24, (intime//60)%60, intime%60, (int((intime - int(intime))*1000)))

    def play(self, fname=None):
        """ff play video

        ffplay -i metro.mov
        ffplay -start_number 5486 -i metro%08d.png
        """
        self._valid(fname)

        _fcmd = [self.ffplay, '-i', self.file]
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

    def playfiles(self, fname=None, folder=".", max_frames=None):
 
        imgs, start = self._io.get_images(folder=folder, name=fname, fmt=('.jpg', '.jpeg', '.png'), max_imgs=max_frames)

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

    def stitch(self, dst, src, audio, fps, size, start_img, max_img):
        """
        ffmpeg -r 29.97 -start_number 5468 -i metro%08d.png -vcodec libx264 -pix_fmt yuv420p /home/z/metropolis_color.mov
        ffmpeg -r 29.97 -start_number 5468 -i metro%08d.png -vcodec libx264 -pix_fmt yuv420p -vframes 200 /home/z/metro_col.mov #only 200 frames
        """
        _ff = 'ffmpeg' if system() != "Windows" else 'ffmpeg.exe'
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

        _fcmd += ['-pix_fmt', 'yuv420p']
        _fcmd += [dst]

        # number of frames # has to be just before outpyut
        if max_img is not None:
            _fcmd += ['-vframes', str(max_img)]

        print(" ".join(_fcmd))
        sp.call(_fcmd)


    def export_frames(self, out_name=None, out_format='.png',
                      start=0, num_frames=1, scale=1):
        """extract frames from video
            fname:
        """

        if out_name is None:
            out_name = osp.splitext(self.file)[0] + self.stats['pad']

        if isinstance(start, float):
            _time = self.strftime(start)
            start = self.time_to_frame(start)
        else:
            _time = self.frame_to_time(start)

        print('exporting frame %s at time %s'%(start, _time))
        _fcmd = [self.ffmpeg, '-i', self.file, '-ss', _time, '-start_number',
                 str(start), '-vframes', str(num_frames)]

        if scale != 1:
            _scale = ['-s', '%dx%d'%(int(self.stats['width'] * scale), int(self.stats['height'] * scale))]
            _fcmd = _fcmd + _scale
            out_name = out_name + '_' + str(scale)

        out_name = out_name + out_format
        _fcmd.append(out_name)
        print(" ".join(_fcmd))
        sp.Popen(_fcmd, stdin=sp.PIPE, stderr=sp.PIPE)

        return osp.abspath(out_name)

    def export_clip(self, out_name=None,
                    start_frame=0, num_frames=0, scale=1):
        """extract video clip
            options:
                formats, [same], '.webm', '.gif'
                scale, [1]
                clip range, [No clipping]

        """
        # auto out name
        if out_name is None:
            out_name = osp.splitext(self.file)[0]+ '_' + str(start_frame) + '-' + str(num_frames+start_frame)
        
        # format

        out_name, out_format = osp.splitext(out_name)
        if not out_format:
            out_format = osp.splitext(self.file)[1]

        # clipping
        if num_frames == 0:
            num_frames = self.stats['frames'] - start_frame
        _time = self.frame_to_time(start_frame,)
        _duration = str(num_frames/self.stats['rate'])

        # prepare subprocess command
        _fcmd = [self.ffmpeg, '-i', self.file]

        # if clip
        if num_frames < int(self.stats['frames']):
            _fcmd = _fcmd + ['-ss', _time, '-t', _duration]

        #if scale
        if scale < 1:
            _scale = ['-s', '%dx%d'%(int(self.stats['width'] * scale), int(self.stats['height'] * scale))]
            _fcmd = _fcmd + _scale
            out_name = out_name + '_' + str(scale)

        #if gif or webm
        if out_format == '.gif' or out_format == '.webm':
            _fcmd = _fcmd + ['-bitrate', '3000k']

        out_name = out_name + out_format
        _fcmd = _fcmd + [out_name]  #, '-hide_banner'
        _p = sp.Popen(_fcmd, stdin=sp.PIPE, stderr=sp.PIPE)
        # for stdout_line in iter(_p.stdout.readline, ""):
        #     yield stdout_line
        # _p.stdout.close()
        # _ret = _p.wait()
        # if _ret:
        #     raise sp.CalledProcessError(_ret, _fcmd)
        return out_name

    def to_numpy(self, start=0, num_frames=None, scale=1):
        """  TODO check it will fit in RAM
        """
        #{'width': 512, 'height': 512, 'rate': 30.0, 'frames': 69, 'pad': '%02d', 'type': 'video', 'file': 'infrerence_6000.mp4'
        out = []

        self._get_stats()
        num_frames = self.stats["frames"] if num_frames is None else min(self.stats["frames"], num_frames + start)

        if isinstance(start, float):
            _time = self.strftime(start)
            start = self.time_to_frame(start)
        else:
            _time = self.frame_to_time(start)

        _fcmd = [self.ffmpeg, '-i', self.file, '-ss', _time, '-start_number', str(start), '-f', 'rawvideo']

        width = self.stats['width']
        height = self.stats['height']
        if scale != 1:
            width = int(self.stats['width'] * scale)
            height = int(self.stats['height'] * scale)
            _scale = ['-s', '%dx%d'%(width, height)]
            _fcmd = _fcmd + _scale
        _fcmd += ['pipe:']

        bufsize = width*height*3

        proc = sp.Popen(_fcmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=bufsize)

        for i in range(start, num_frames):
            buffer = proc.stdout.read(bufsize)
            if len(buffer) != bufsize:
                break
            out += [np.frombuffer(buffer, np.uint8).reshape(height, width, 3)]
        out = np.stack(out, axis=0)
        proc.stdout.close()
        proc.wait()
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

#get
#sexagesimal file duration
ffprobe -v error -show_entries format=duration -sexagesimal -of default=noprint_wrappers=1:nokey=1 MUCBCN.mp4

#millisecond file duration
ffprobe -v error -show_entries format=size -of default=noprint_wrappers=1:nokey=1 MUCBCN.mp

#frame size
ffprobe -v error -of flat=s=_ -select_streams v:0  -show_entries stream=height,width MUCBCN.mp4
#frame size, no key
ffprobe -v error -of default=noprint_wrappers=1:nokey=1 -select_streams v:0  -show_entries stream=height,width MUCBCN.mp4

#frame count
ffprobe -v error -of default=noprint_wrappers=1:nokey=1 -select_streams v:0 -show_entries stream=nb_frames MUCBCN.mp4

#framerate
ffprobe -v error -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 MUCBCN.mp4
25/1 -> is pal
0/0

ffprobe -v error -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 MUCBCN.mp4

#bunch of info
ffprobe -i MUCBCN.mp4 -show_stream
ffprobe -i MUCBCN.mp4 -show_format

#play
#half speed (audio doen not chnage speed)
ffplay MUCBCN.mp4 -vf "setpts=2*PTS"

"""