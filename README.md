# VIDI
=====

### WIP: ffmpeg wrapper used on Ubuntu only
### Not sufficiently robust, untested
rewrite to simplify
* plays video
* stitch video from frames
* stream from numpy to video 
* convert video to numpy

requires ffmpeg
```bash
conda install ffmpeg
```
depending on the verstion there can be a mismatch in libraries leading to error

`ffprobe: error while loading shared libraries: libopenh264.so.5: cannot open shared object file: No such file or directory`

That error can be fixed by creating a symlink from an existing `libopenh264.so`, e.g.

`ln -s libopenh264.so libopenh264.so.5`



# Examples:

## Play files
play video files with [ffplay](https://ffmpeg.org/ffplay.html)<br>
`vidi.ffplay(src, start=0, fps=None, loop=0, autoexit=True, fullscreen=False, noborder=True)`
```python
import vidi
# play video file
vidi.ffplay('metro_color_tb4.mov') 
# play images in folder
vidi.ffplay("/home/z/myfolder/*.png")
# play images with pattern in current folder
vidi.ffplay("image%06d.jpg", loop=2, noborder=False,)
# show frame numbers in yellow
vidi.ffplay('metro_color_tb4.mov', showframe=True, fontcolor="yellow") 
```

## Stitch files to video
```python
# stitch 2000 images with template, starting at frame 6000into an .mov, add audio
vidi.ffstitch("image%08d.png", "mymovie.mov", audio="/home/z/data/Music/mymuzak.aac", start=6000, num=2000)
# stitch 100 images with template into an .mov of size 50x50
vidi.ffstitch("metro%08d.png", "mymovie.mov", num=100, size=50)
# stitch all .pngs in folder, into an .mov of size w:1000, h:200
vidi.ffstitch("*.png", "mymovie.mov", size=(1000,200))
```

## ndarray to video
```python
with vidi.FFcap('myvideo.mp4', size=(height, width), fps=30) as F:
    F.add_frame(ndarray)
# ndarray of shape (height, widht, 3) or (nb_frames, height, width, 3)
# only tested with filenames in .mp4, .avi, pix_fmt='rbg24' and 'gray'
```

## video to ndarray
```python
out = vidi.ffread(<videofile>) <br>

# alternate method, n frames
f = vidi.FF(<videofile>)
imgs = f.to_numpy(start=0, nb_frames=None, scale=1, stream=0, step=1, dtype=np.uint8, memory_type="CPU")
# returns imgs of shape (nb_frames, height, width, channels)
```
## show frame with matplotlib
```python
f = vidi.FF(<videofile>)
f.view_frame(<time in sec. | framenumber>)
```

## export video clip
```python
f = vidi.FF(<videofile>)
f.export_clip(out_name="myclip.mp4", num_frames=4, scale=0.5)
```

## export video frames
```python
f = vidi.FF(<videofile>)
f.export_frames(out_name="myclip_%06d.png", start=0, nb_frames=4, scale=0.5, step=1, stream=0, out_folder=None)
```

## video information
```python
stats = vidi.ffprobe(<videofile>, verbose=False, entries=None)
```

## export frame and time to .srt subtitle file
```python
f = vidi.FF(<videofile>)
f.make_subtitles()
```