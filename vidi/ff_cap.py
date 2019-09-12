""" capture utility
"""
import platform
import os.path as osp
import subprocess as sp
import numpy as np
DEBUG = False




class FFcap:

    def __init__(self, name='vid.avi', size=(640, 480), fps=30, increment=True, overwrite=True,
                 codec="raw", pix_fmt="rgb24", debug=False):
        """
        Args
            name        name of output video
            size        tuple (width, height) [640,480]
            fps         int, float [30]
            increment   bool [True], early closure does not corrupt file
            overwrite   bool [True],  overwrite file if found
            codec       str [h264], "h264" or "raw"
            pix_fmt     str [yuv420p], 'rgb', 'gray'
        """


        self.debug = debug
        self.name = name

        # options
        self.size = (size, size) if isinstance(size, int) else size
        self.fps = fps

        self.increment = increment
        self.overwrite = overwrite
        self.codec = codec
        self.pix_fmt = pix_fmt

        #self.codec
        self.audio = False

        self.ffmpeg = 'ffmpeg' if platform.system() != 'Windows' else 'ffmpeg.exe'
        self._pipe = None
        self._framecount = 0

    # def __enter__(self):
    #     print('entering, fail()')
    #     return self

    # def __exit__(self, exc_type, exc_value, traceback):
    #     self.close()

    # def __del__(self):
    #     self.close()

    # def concat(self, files):

    #     self.ffmpeg
    #     cmd = [self.ffmpeg]
    #     if self.overwrite:
    #         cmd += ['-y']

    #     cmd += ['-i']


    def from_frames(self):
        """given image frames
        """
        cmd = [self.ffmpeg]
        if self.overwrite:
            cmd += ['-y']

        #vlc unsupported codec 28 or profile 244
        # source video chroma type not supported

        # if self.codec == "raw":
        cmd += ['-f', 'rawvideo']
        cmd += ['-vcodec', 'rawvideo']
        # else:
        #     cmd += ['-vcodec', 'libx264']


        #-vcodec libx264 -pix_fmt yuv420p

        # image size: if image size != frame size given, this will fail
        cmd += ['-s', '%dx%d'%self.size]
        cmd += ['-pix_fmt', self.pix_fmt]

        # frames per second in resulting video
        cmd += ['-r', str(self.fps)]

        # from stream
        cmd += ['-i', '-']

        # audio
        if not self.audio:
            cmd += ['-an']
        else:
            #check, record desktop or record mic
            cmd += ["-thread_queue_size", "1024"]
            cmd += ["-f", "alsa", "-ac", "2", "-i", "pulse"]

        if self.increment:
            cmd += ["-movflags", "frag_keyframe"]

        cmd += [self.name]
        if self.debug:
            print('  DEBUG: <ff.py>: ff.FF.from_frames() cmd: \n\t%s'%" ".join(cmd))

        self._pipe = sp.Popen( cmd, stdin=sp.PIPE, stderr=sp.PIPE)
        if self.debug:
            print('  DEBUG: <ff.py>: ff.FF.from_frames() FF._pipe: \n\t%s'%self._pipe)
            print('  DEBUG: <ff.py>: ff.FF.from_frames() FF._pipe.stderr: \n\t%s'%self._pipe.stderr)
            print('  DEBUG: <ff.py>: ff.FF.from_frames() FF._pipe.stderr: \n\t%s'%self._pipe.stderr)

    def add_frame(self, frame):
        #debugprint
        if self.debug:
            print('  : <ff.py>: ff.FF.add_frame(frame): %d, %s'%(self._framecount, str(frame.shape)), end="\r")

        self._pipe.stdin.write(frame.tobytes())
        self._framecount += 1

    def from_screen(self):
        pass

    def close(self):
        if self._pipe is not None:
            self._pipe.stdin.close()
            if self._pipe.stderr:
                self._pipe.stderr.close()
        print('FF.close()  <ff.py> ')
        print('  recorded to file <%s>, Exists <%s>; nbframes <%d>'%(self.name,
                                                                     str(osp.isfile(self.name)),
                                                                     self._framecount), end="\r")
