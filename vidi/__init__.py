""" wrapper to ffmpeg video
"""
from .ff_main import FF, FFDataset

# from .io_main import IO
# from .ff_dataset import AVDataset

# from .ff_cap import FFcap


# TODO fix torch video dataset with augmentation
# from .ff_dataset import AVDataset

# TODO add stack
#ffmpeg -i vid_l.mp4 -i vid_r.mp4 -filter_complex hstack -c:v libx264 out.mp4


#pylint: disable=import-error
from .version import version as __version__
