"""vidi main,  file for motion pose video analysis"""

import os
import os.path as osp
from .io_main import IO
from .ff_main import FF
# from .cv_main import CV
# from .annotator import Annotator

MAIN_WIN = 'VIDI'

class VIDI:
    """V class, main motion pose video analysis
        fname:  video file name str
                can be inexistent yet if for recording
                if None, it must be input in play or stitch or record

        backend 'ff'|'cv'



        # stitch all pngs to an avi
        ffmpeg -pattern_type glob -i '*.png' out.avi

        # play all pngs
        ffplay -pattern_type glob -i 'metro*.png'

        # stitch all files called metro%08d.png to a mov
        ffmpeg -r 29.97 -start_number 5468 -i metro%08d.png -vcodec libx264 -pix_fmt yuv420p /home/z/metropolis_color.mov

        #stitch 200 frames
        ffmpeg -r 29.97 -start_number 5468 -i metro%08d.png -vcodec libx264 -pix_fmt yuv420p -vframes 200 /home/z/metro_col.mov #only 200 frames
    """

    def __init__(self, fname=None):#, annotation_file=None, backend="ff"):

        self.fname = fname # name of video, existing or not
        self.io = IO()
        self.ff = FF(self.fname)
        self.backend = 'ff'

        self.image_fmt = ['.jpg', '.jpeg', '.png']
        self.video_fmt = ['.mkv', '.avi', '.mp4', '.mov', '.flv']

        self.cap = None
        self.template = None
        self.a_file = None
        self.vid = None

    def find(self, folder='.', ext=None):
        if isinstance(ext, str):
            ext = [ext]
        if ext is None:
            ext = self.video_fmt
        files = [f for f in os.listdir(folder) if osp.splitext(f)[1] in ext]
        if folder != '.':
            files = [osp.join(folder, f) for f in files]
        return files

    def play(self, fname=None, folder=None, max_time=None):
        """
        Args
            fname, image sequence or video
            folder, optional
            max_time, if image sequence is frame, if video is time

        Examples
        >>> V = VIDI('/home/z/metro_color.mov')
        >>> V.play()
        """
        if (fname is None and folder is not None) or (fname is not None and '%' in fname):
            return self.ff.playfiles(fname=fname, folder=folder, max_frames=max_time)
            
        if fname is not None:
            self.fname = fname

        # file io is tied in knots. ridiculous
        self.io.file_resolve(self.fname, folder)

        self.__dict__[self.backend].play(self.io.file)

    def export_frames(self, fname=None, out_name=None, out_format='.png',
                      start=0, num_frames=1, scale=1):
        if fname is not None:
            self.fname = fname
        self.io.file_resolve(self.fname)

        # if self.backend != 'ff':
        #     print("using ffmpeg: frame export currently only available in ffmpeg")
        #     self.set_backend('ff')

        self.ff.export_frames(fname, out_name, out_format, start, num_frames, scale)


    def stitch(self, src, name=None, folder='.', audio=None, fmt=".png",
               fps=29.97, size=None, start_img=None, max_imgs=None):
        """
        Examples
        >>> V = VIDI()
        >>> dst = '/home/z/metro_color.mov'
        >>> input_folder = '/home/z/work/gans/pix2pix/results/color_charlie_xavier_rnd'
        >>> src = 'metro%08d.png'
        >>> start_frame = 5468
        >>> V.stitch(src, name=dst, folder=input_folder, start_img=start_frame)
        >>> V.play()
        """

        assert "%" in src, "source name needs to be teplate name e.g. metro%06d.png, got <%s%"

        #src, start_img = self.io.get_images(folder, src, fmt, max_imgs)
        folder = osp.abspath(folder)

        if name is None:
            name = src.split("%")[0]+".mov"

        # if not osp.isdir(osp.split(name)[0]):
        #     name = osp.join(folder, name)
        print("--------Stitching file ------\n", name)

        _cwd = os.getcwd()
        try:
            os.chdir(folder)
            self.ff.stitch(name, src, audio, fps, size, start_img, max_imgs)
            self.file = name
        except:
            print("failed to stitch video")
        os.chdir(_cwd)

    # def load_annotation(self, afile=None):
    #     # TODO to do allow loading other formats
    #     if osp.isfile(afile):
    #         self.a_file = afile
    #         A = Annotator()
    #         A.load(self.a_file)
    #         self.io.template = A.A

    