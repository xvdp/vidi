"""@xvdp
"""
from vidi.utils import frame_to_time, frame_to_strftime, time_to_frame, strftime, strftime_to_time, anytime_to_frame_time

def test_frame_to_time():
    assert frame_to_time(24, 24) == 1.0
    assert frame_to_time(60, 30) == 2.0
    assert frame_to_time(0, 10) == 0.0

def test_frame_to_strftime():
    assert frame_to_strftime(frame=0, fps=30) == "00:00:00.000"
    assert frame_to_strftime(frame=90, fps=30) == "00:00:03.000"
    assert frame_to_strftime(5400, fps=30) == "00:03:00.000"

def test_time_to_frame():
    assert time_to_frame(1.0, 24) == 24
    assert time_to_frame(2.0, 30) == 60
    assert time_to_frame(0.0, 10) == 0

def test_strftime():
    assert strftime(0) == "00:00:00.000"
    assert strftime(90) == "00:01:30.000"
    assert strftime(5400) == "01:30:00.000"

def test_strftime_to_time():
    assert strftime_to_time("00:00:00.000") == 0.0
    assert strftime_to_time("00:01:30.000") == 90.0
    assert strftime_to_time("01:30:00.000") == 5400.0

def test_anytime_to_frame_time():
    assert anytime_to_frame_time(24, 24) == (24, 1.0)
    assert anytime_to_frame_time(2.0, 30) == (60, 2.0)
    assert anytime_to_frame_time("00:01:30.000", 30) == anytime_to_frame_time(90., 30) == (2700, 90.0)

