""" Input Output for Vidi project

Cleanup!!!
"""
import os
import os.path as osp

class IO:
    """ Input Output class for Vidi project"""
    def __init__(self, fname=None, template=None):
        self.file = None
        self.filetype = None
        #self.annotation = None

        self.file_resolve(fname)
        self.template = template

    def get_images(self, folder='.', name=None, fmt=('.jpg', '.jpeg', '.png'), max_imgs=None, as_str=False):
        """ Returns list or concat str of files

            folder   folder to find images
            name     list, name template (e.g. out%08d.png ), None, if none, find in folder
            fmt      list or tuple of valid image extensions
            max_imgs int | [None]
            as_str   bool [False]
        """
        start_frame = None
        folder = osp.abspath(folder)
        assert osp.isdir(folder), "<%s> not a valid folder"
        if name is None:
            name = sorted([f for f in os.listdir(folder) if osp.splitext(f)[1].lower() in fmt])

        if isinstance(name, list):
            name = name[:max_imgs]

        if isinstance(name, list):
            name = [osp.join(folder, f) for f in name]
            if name and as_str:
                name = 'concat:"' + '|'.join(name) +'"'

        elif isinstance(name, str):
            assert '%' in name, "template name not recognized expected format name%08d.png"
            if not osp.isdir(osp.split(name)[0]):
                name = osp.join(folder, name)
            
            name_root = osp.basename(name.split("%")[0])
            fmt = osp.splitext(name)[1]
            #print(name_root, fmt, name)
            first_file = sorted([f for f in os.listdir(folder) if osp.splitext(f)[1].lower() in fmt and name_root in f])[0]

            start_frame = int(osp.splitext(first_file.split(name_root)[1])[0])
            
        else:
            print("name must be None, template string or list, found ", type(name))
            name = False
        
        return name, start_frame


    def file_resolve(self, fname=None, folder=None):
        """ failure tolearant file resolver
        TODO: add image type support
        TODO: replace into FF
        """
        # only clobber self.file is fname exists
        if fname is not None:
            if folder is not None:
                fname = osp.join(folder, fname)
            assert os.path.isfile(fname), "requested file not found <%s>"%fname
            self.file = fname

        # validate existing file, set to None if not found
        if self.file is None or not os.path.isfile(self.file):
            self.file = None

        # TODO ensure file is of correct set of types
        video = ['.mp4', '.avi', '.flv', '.mov']
        image = ['.jpg', '.png']
        database = ['.hdf5']

        return self.file

    def file_valid(self, fname=None):
        """ assert if no valid file found """
        self.file = self.file_resolve(fname)
        assert self.file is not None, 'enter valid file'
        return self.file


    def get(self, fname=None):
        """get video file name"""
        self.file = fname
        if self.file is None or not os.path.isfile(self.file):
            self.file = self.default()

        assert os.path.isfile(self.file)
        return self.file

    def get_img_pair(self, fname=None, start_frame=0):
        """extracts two images out of a video file"""
        self.get(fname)

    def default(self):
        """get default file name"""
        roots = ['/users/jvdp/work/Data', '/home/z/work/Data']
        fname = r'Foot/fromwild/videos/'
        #bname = r'Ronaldo goal in 2002 World Cup Final - 1080p HD.mp4'
        bname = r'MUCBCN.mp4'
        for root in roots:
            full_fname = os.path.join(root, fname, bname)
            if os.path.isfile(full_fname):
                return full_fname


    def _get_ui(self):
        pass
    def _get_console(self):
        pass
    def _store_json(self):
        pass
    def _load_json(self):
        pass