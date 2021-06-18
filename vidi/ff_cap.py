""" capture utility
"""
import platform
import subprocess as sp
from .utils import Col, dprint

class FFcap:

    def __init__(self, name='vid.avi', size=(640, 480), fps=30, increment=True, overwrite=True,
                 pix_fmt="rgb24", src_type="stdin", debug=False):
        """
        Args
            name        name of output video
            size        tuple (width, height) [640,480]
            fps         int, float [30]
            increment   bool [True], early closure does not corrupt file
            overwrite   bool [True],  overwrite file if found

            pix_fmt     str ['rgb24'], 'rgb24', 'gray' 
                # should pass any of `ffmpeg -pix_fmts`  but #'yuv420p' does not work


            src_type      str ['stdin'] 'stdin' | (not implemented): 'screen'| webcam | ximea

        Example:

            with vidi.FFcap("myvid.mp4", pix_fmt='rgb24, fps=29.97, size=(640,480), overwrite=True, debug=True, src_type='stdin') as F:
                F.add_frame(ndarray)
        """


        self.debug = debug
        self.name = name

        # options
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        assert len(self.size) == 2, "size must be a (width, height) tuple, got <%s> len %d"%(str(type(self.size)), len(self.size))
        self.fps = fps

        self.increment = increment
        self.overwrite = overwrite
        assert pix_fmt in ("rgb24", "gray")
        self.pix_fmt = pix_fmt
        self._channels = 3 if pix_fmt == "rgb24" else 1
        self._shape = self.size + (self._channels,)
        assert src_type in ("stdin"), NotImplementedError
        self.src_type = src_type

        self.audio = False

        self._cmd = None
        self._ffmpeg = 'ffmpeg' if platform.system() != 'Windows' else 'ffmpeg.exe'
        self._pipe = None
        self._framecount = 0


    def __enter__(self):
        dprint('%sFFcap.__enter__()\n%s'%(Col.YB, Col.AU), debug=self.debug)
        self.open()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        dprint('%sFFcap.__exit__()\n%s'%(Col.YB, Col.AU), debug=self.debug)
        self.close()


    def open(self):
        if self._cmd is None:
            if self.src_type == "stdin":
                self.init_frames()
        if self._pipe is not None:
            self.close()
        dprint('%sOpening PIPE%s'%(Col.GB, Col.AU), debug=self.debug)
        self._pipe = sp.Popen(self._cmd, stdin=sp.PIPE, stderr=sp.PIPE)

    def close(self):
        if self._pipe is not None:
            dprint('%sClosing PIPE%s'%(Col.YB, Col.AU), debug=self.debug)
            
            self._pipe.stdin.flush()
            self._pipe.stdin.close()
            self._pipe.stderr.close()
            self._pipe.terminate()
            # self._pipe.kill()
        self._pipe = None
        dprint('%sPIPE Closed%s'%(Col.YB, Col.AU), debug=self.debug)

    def init_frames(self):
        """given image frames
        """
        self._cmd = [self._ffmpeg]
        if self.overwrite:
            self._cmd += ['-y']

        #vlc unsupported codec 28 or profile 244
        # source video chroma type not supported

        # if self.codec == "raw":
        self._cmd += ['-f', 'rawvideo']
        self._cmd += ['-vcodec', 'rawvideo']
        # else:
        #     self.increment = False
        #     self._cmd += ['-vcodec', 'libx264']

        # image size: if image size != frame size given, this will fail
        self._cmd += ['-s', '%dx%d'%self.size]
        self._cmd += ['-pix_fmt', self.pix_fmt]

        # frames per second in resulting video
        self._cmd += ['-r', str(self.fps)]

        # from stream
        self._cmd += ['-i', '-']

        # audio
        if not self.audio:
            self._cmd += ['-an']
        else:
            #check, record desktop or record mic
            self._cmd += ["-thread_queue_size", "1024"]
            self._cmd += ["-f", "alsa", "-ac", "2", "-i", "pulse"]

        if self.increment:
            self._cmd += ["-movflags", "frag_keyframe"]

        self._cmd += [self.name]
        dprint('%s%s%s'%(Col.GB, (" ".join(self._cmd)), Col.AU), debug=self.debug)


    def add_frame(self, frame):
        """ pix_fmt rgb24 requires uint8 RGB
        """
        if self.debug:
            print('%sadd_frame() %d %s%s'%(Col.BB, self._framecount, str(frame.shape), Col.AU), end="\r")
        assert frame.shape == self._shape, "attempting to input incorrect file size, <%s> instead of <%s>"%(str(frame.shape), str(self._shape))
        self._pipe.stdin.write(frame.tobytes())
        self._framecount += 1

    def from_screen(self):
        pass
