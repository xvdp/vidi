# VIDI 
## Python wrapper to FFMPEG
v.0.2 wipe simplified and rewritten
*   Linux only (for windows replace 'ffmpeg' with 'ffmpeg.exe' in vidi.ff.py - untested)

Exposes a subset of ffmpeg functionality wrapped in functions. VIDI's main intention is interaction between video and array data, numpy and torch.

Converts RGB to and from FourCC and higher bits formats 10,12,14 and 16 bit files.

---

### classes

**`vidi.FF`** # requires numpy, cv2

    V = FF(videofile)
    V.export_clip()
    V.export_frames()
    V.to_numpy()        # -> ndarray
    V.make_subtitles()  # frame number and time subtitles
    V.vlc()             # play with vlc
    V.play()            # ffplay
    V.stats             # ffprobe -> dict

**`vidi.FFDataset`**

    with FFDataset(videofile) as D:
        D.__getitem__() # -> torch.tensor if torch installed else np.ndarray
        D.__len__()


### standalone functions

    make_subtitles(videofile)   # creates subtitle file with frame numbers
    export_frames(videofile)    # exports frame range to images
    export_clip(videofile)
    get_formats()               # IO vidi subset of installed ffmpeg supported video formats
    images_to_video()           # creates video from images
    yxx2rgb()                   # FourCC (YCC and YUV) to RGB
    rgb2yxx()                   # RGB to FourCC (YCC and YUV)
    yxx_matrix()
    expand_fourcc() 4nm -> 444  # uncompress raw FourCC flat arrays to 3 channels
    # todo expose compress_fourcc()

### TODO: 
* unit tests; 
* multiprocessing to torch dataset