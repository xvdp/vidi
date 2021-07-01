#!/bin/bash
## combine renders and add fonts
# TODO add function to vidi?

# concatenate
LEFT=cat_recon_5000.mp4
MID=cat_recon_13000.mp4
RIGHT=/media/z/Malatesta/zXb/share/siren/data/cat_video.mp4
COMP=cat_5_13K.mp4

ffmpeg -i $LEFT -i $MID -filter_complex hstack -c:v libx264 $COMP
ffmpeg -i $COMP -i $RIGHT -filter_complex hstack -c:v libx264 $COMP

# add fonts
# get fonts: fc-list
FONT=/usr/share/fonts/truetype/ubuntu/UbuntuMono-RI.ttf
TEXT='5k iters lr1e-4 loss 0.0042              +8k iters lr3.16e-5 loss 0.0022                 original'
ANNOT=cat2_5_13K.mp4
ffmpeg -i $COMP -c:v libx264 -vf drawtext="fontfile=$FONT:text=$TEXT: fontcolor=white: fontsize=24: x=(w-text_w)/4: y=(h-text_h)/8" $ANNOT

# render text and combined video
ffplay -i $ANNOT
