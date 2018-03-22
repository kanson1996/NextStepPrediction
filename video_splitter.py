#!/usr/bin/python3
# -*- coding: utf-8 -*-
from moviepy.editor import VideoFileClip
from argparse import ArgumentParser
from os import path
import numpy as np


def detect_cut(frame1, frame2, threshold=320.0):
    mean_absolute_distance = np.sum(np.absolute(frame1 - frame2)) / (frame2.shape[0] * frame2.shape[1])
    print(mean_absolute_distance)
    return mean_absolute_distance > threshold


def split_video(filename, destination_folder, min_clip_len=10):
    source = VideoFileClip(filename)
    if not source.fps:
        source.set_fps(25)
    last_frame = source.get_frame(0.0)
    last_cut = 0.0
    clip_counter = 0
    for time, frame in source.iter_frames(with_times=True, dtype=np.uint8):
        if detect_cut(last_frame, frame):
            if (time - last_cut) * source.fps >= min_clip_len:
                clip = source.subclip(last_cut + 1 / source.fps, time)
                clip.write_videofile(
                    path.join(destination_folder,
                              "{}-{}.mp4".format(path.split(filename)[-1].split('.')[0], clip_counter)),
                    audio=False, codec="libx264", bitrate='2000000')
                clip_counter += 1
            last_cut = time
        last_frame = frame


if __name__ == "__main__":
    parser = ArgumentParser("Split an edited video in its individual shots and save the clips as training data.")
    parser.add_argument('filename', type=str, help="The filename of the input video.")
    parser.add_argument('-d', '--destination', type=str,
                        help="Destination folder for the clips. Default: training_data"+path.pathsep,
                        default="training_data")
    args = parser.parse_args()
    split_video(args.filename, args.destination)
