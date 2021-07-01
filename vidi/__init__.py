""" wrapper to ffmpeg video
"""
from .io_main import IO
from .ff_main import FF

from .ff_fun import *
from .ff_cap import FFcap


# TODO fix torch video dataset with augmentation
# from .ff_read import FFread
# from .ff_dataset import AVDataset

# TODO add stack

#ffmpeg -i vid_l.mp4 -i vid_r.mp4 -filter_complex hstack -c:v libx264 out.mp4


#pylint: disable=import-error
from .version import version as __version__
