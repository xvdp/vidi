""" capture utility
"""
import platform
import subprocess as sp

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
            pix_fmt     str ['rgb24'], 'rgb24', 'gray' # not handled: yuv420p
            src_type      str ['stdin'] 'stdin' | (not implemented): 'screen'| webcam | ximea
                # should pass any of `ffmpeg -pix_fmts`

        Example:
            with vidi.FFcap(name+ext, pix_fmt='rgb24, fps=29.97, size=(640,480), overwrite=True, debug=True, src_type='stdin') as F:
                F.init_frames()
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
        print('\n\t .__enter__()')
        self.open()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.debug:
            print('\n\t .__exit__()\n')
        self.close()


    def open(self):
        if self._cmd is None:
            if self.src_type == "stdin":
                self.init_frames()
        if self._pipe is not None:
            self.close()
        self._pipe = sp.Popen(self._cmd, stdin=sp.PIPE, stderr=sp.PIPE)

    def close(self):
        if self._pipe is not None:
            self._pipe.stdin.close()
        self._pipe = None

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
        if self.debug:
            print('  DEBUG: <ff.py>: ff.FF.init_frames() cmd: \n\t%s'%" ".join(self._cmd))


    def add_frame(self, frame):
        #debugprint
        if self.debug:
            print('  : <ff.py>: ff.FF.add_frame(frame): %d, %s'%(self._framecount, str(frame.shape)), end="\r")
        assert frame.shape == self._shape, "attempting to input incorrect file size, <%s> instead of <%s>"%(str(frame.shape), str(self._shape))
        self._pipe.stdin.write(frame.tobytes())
        self._framecount += 1

    def from_screen(self):
        pass
