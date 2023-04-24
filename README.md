# VIDI Wrapper to FFMPEG for python / Linux
v.0.2 wipe simplified and rewritten
*   Linux only (for windows replace 'ffmpeg' with 'ffmpeg.exe' in vidi.ff.py - untested)
*   requires ffmpeg installed in system

Rewrittien to handle FourCC including correctly reading and higher bits formats 10,12,14 and 16 bit files. FourCC formats can be read 

---

ffmpeg wrpper classes

**`vidi.FF`** # requires numpy, cv2

    V = FF(videofile)
    V.export_clip()
    V.export_frames()
    V.to_numpy()        # -> ndarray
    V.make_subtitles()  # frame number and time subtitles
    V.vlc()             # play with vlc
    V.play()            # ffplay
    V.stats             # ffprobe -> dict

**`vidi.FFDataset`** # requires pytorch

    with FFDataset(videofile) as D:
        D.__getitem__() # -> torch tensor
        D.__len__()


**`standalone functions`**

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

### TODO: add unit tests / add multiprocessing FFDataset and test.