# VIDI Wrapper to FFMPEG for python / Linux

*   Linux only (for windows replace 'ffmpeg' with 'ffmpeg.exe' in vidi.ff.py - untested)
*   requires ffmpeg installed in syste,
*   v.0.2 wipe simplify and rewrite

TODO: tests
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

