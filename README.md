# VIDI
=====

unsupported ffmpeg wrapper to:
* play video
* stitch video from frames
* stream from array to video

requires
```bash
sudo apt-get install ubuntu-restricted-extras
sudo apt-get install libavcodec-dev libav-tools ffmpeg
```

# Examples:

## Play files
```python
import vidi
# play video file
vidi.ffplay('metro_color_tb4.mov') 
# play images in folder
vidi.ffplay("*.png", folder="/home/z/myfolder/")
# play images with pattern in current folder
vidi.ffplay("image%06d.jpg")
```

## Stitch files to video
```python
# stitch 2000 images with template, starting at frame 6000into an .mov, add audio
vidi.ffstitch("image%08d.png", "mymovie.mov", audio="/home/z/data/Music/mymuzak.aac", start=6000, num=2000)
# stitch 100 images with template into an .mov of size 50x50
vidi.ffstitch("metro%08d.png", "mymovie.mov", num=100, size=50)
# stitch all .pngs in folder, into an .mov of size w:1000, h:200
vidi.ffplay("*.png", folder="~/work/gans/results", size=(1000,200))
```

## ndarray to video
```python
with vidi.FFcap('myvideo.mp4', size=(256,256)) as F:
    F.add_frame(ndarray)
# only tested with filenames in .mp4, .avi, pix_fmt='rbg24' and 'gray'
```
