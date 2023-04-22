""" @xvdp
Simple solution  to Show frame number and frame time: Make a subtitle file
run this file on the video, tehn 

To play with suptitles:
$ vlc <video_file> 
or
$ ffplay -vf subtitles=<subtitles.srt> <video_file> 


## another way to print frame numbers is to utilize ffplay drawtext filter but it returns local frame numbers, not global.
they get reset on fast forward
as .e.g
#
$ ffplay -vf drawtext=fontfile=Arial.ttf:x=w*0.01:y=h-2*th:fontcolor=white:fontsize=h*0.0185:text='%{n}' <video_file>

start_number:

other useful ffplay options - notes

-fs Start in fullscreen mode.
-ss pos
-t duration in seconds

-noborder


s   Step to the next frame.
p, spacebar  Pause
f   Toggle full screen.
q
"""

import sys
import os
import os.path as osp
import json


def frame_subtitles(filename, stream=0):
    """ creates frame number and frame time from video
    """
    nb_frames, frame_rate = ffmpeg_stats(filename, stream)
    make_subtitle(filename, nb_frames, frame_rate, stream)

def make_subtitle(filename, nb_frames, frame_rate, stream=0):
    """ 
    nb_frames and some form of frame_rate you can get from ffprobe
    depending on how video was recorded there may be an avg_frame_rate and DURATION pr...
    """
    fname = osp.abspath(filename)
    assert osp.isfile(fname), f"file not found {fname}"
    name = f"{osp.splitext(fname)[0]}_frames.srt"
    print(f"Making Frame / Frame time subtitle file {filename} stream [{stream}] -> {name}")

    sub = ""
    last_frame = frame_to_strftime(0, frame_rate)
    for i in range(nb_frames):
        next_frame = frame_to_strftime(i+1, frame_rate)
        sub += f"{i+1}\n{last_frame} --> {next_frame}\n\t{i+1}\t{last_frame}\n\n"
        last_frame = next_frame
    with open(name, 'w', encoding='utf8') as _fi:
        _fi.write(sub)


def ffmpeg_stats(filename, stream=0):
    """ return (nb_frames, frame_rate) from video filename
    """
    assert osp.isfile(filename), f"{filename} not a valid file"
    _cmd = f"ffprobe -v quiet -print_format json -show_format -show_streams {filename}"
    with os.popen(_cmd) as _fi:
        stats = json.loads(_fi.read())

    if 'streams' in stats:
        videos = [s for s in stats['streams'] if s['codec_type'] == 'video']
    assert videos is not None, f'no video streams found in {filename}'

    if len(videos) > 1:
        print(f"video {filename} cotains {len(videos)} streams, processing stream={stream}")
    _stats = videos[stream]

    _frame_rate_keys = [k for k in _stats if 'frame_rate' in k]
    assert len(_frame_rate_keys), f"no frame_rate keys found in  {filename}\n{json.dumps(_stats)}"
    frame_rate = eval(_stats[_frame_rate_keys[0]])

    if 'nb_frames' in _stats:
        nb_frames = eval(_stats['nb_frames'])
    elif 'tags' in _stats and 'DURATION' in _stats['tags']:
        duration = _stats['tags']['DURATION']
        seconds = sum(x * float(t) for x, t in zip([3600, 60, 1], duration.split(":")))
        nb_frames = round(seconds * frame_rate)
    else:
        print(f"neither 'nb_frames' and 'DURATION' found in stream\{json.dumps(_stats)}")
        return None

    return nb_frames, frame_rate

def frame_to_strftime(frame: int = 0, frame_rate: float = 30.) -> str:
    """convert frame number to time"""
    _tf = frame/frame_rate
    _ti = int(_tf)
    return f"{(_ti//3600)%24:02d}:{(_ti//60)%60:02d}:{_ti%60:02d}.{(int((_tf - _ti)*1000)):03d}"


if __name__ == "__main__":
    assert len(sys.argv) > 1, "usage: $ python frame_subtitles.py <video_file>"
    frame_subtitles(sys.argv[1])
