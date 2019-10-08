# VIDI
=====

Examples:
* accessing through vidi 

## Classes
* AVDataset(Dataset)
    `AV.__getitem__()`
    file: ff_dataset.py
    examples: vidi/jupyter/AVDataset.ipynb
* FFread()
    ff_read.py
* FFcap()
    ff_cap.py

## Functions
* `ffprobe(src)`
* `ff_play(src, folder=None, start=0, fps=None, loop=0, autoexit=True, fullscreen=False)`
* `ffstitch(src, dst, folder=None, fps=29.97, start=0, size=None, num=None, audio=None)`
    ff_fun.py
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

## Stitch files to movie
```python
# stitch 2000 images with template, starting at frame 6000into an .mov, add audio
vidi.ffstitch("image%08d.png", "mymovie.mov", audio="/home/z/data/Music/mymuzak.aac", start=6000, num=2000)
# stitch 100 images with template into an .mov of size 50x50
vidi.ffstitch("metro%08d.png", "mymovie.mov", num=100, size=50)
# stitch all .pngs in folder, into an .mov of size w:1000, h:200
vidi.ffplay("*.png", folder="~/work/gans/results", size=(1000,200))
```

```python
vidi.ffplay('metro_color_tb4.mov')
```


# TODO REVISE VALIDITY
--------------------
```python
    import os
    import vidi

    f=os.path.expanduser('~/work/Data/Foot/fromwild/videos/MUCBCN.mp4')
    
    v = vidi.V(f) 

    v.cv.get_stats()
    v.ff.get_stats()

    v.ff.play()
    v.cv.play()

    #Example 2, load a different file
    g=os.path.expanduser('~/work/Data/Foot/fromwild/videos/Ronaldo goal in 2002 World Cup Final - 1080p HD.mp4')
    v.setfile(g)

    #Example 3, load also annotation
    a ='/home/z/work/pneumonia/extra.json'
    v = vidi.V(f, a)
    v.cv.play()

```
* annotation
```python
    #creates annotation template
    from vidi import Annotator
    A = Annotator()
    A.marker('L4Root')
    ...
    A.line('Spine')
    A.write('/home/z/work/pneumonia/extra.json')
    print(A)

    #loads annotation template
    from vidi import Annotator
    A = Annotator()
    A.load('/home/z/work/pneumonia/extra.json')
    print(A)
```

cv
--
Keystrokes:
    'q'     quit
    'space' pause
    'm'     start marker
    '='     faster
    '-'     slower
    '0'     normal speed

Requirements
------------
python > 3.4
opencv > 3.4
pip install cvui https://github.com/Dovyski/cvui

