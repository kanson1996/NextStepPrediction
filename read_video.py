#!/usr/bin/python3
# -*- coding: utf-8 -*-
from keras.utils import Sequence
from moviepy.editor import VideoFileClip
from skimage.transform import resize
import numpy as np


class VideoTooShortException(Exception):
    pass


class VideoReader(Sequence):
    def __init__(self, file_list=None, seq_len=10):
        self.video_files = file_list if file_list else []
        self.seq_len = seq_len

    def __len__(self):
        return np.ceil(len(self.video_files) / float(self.seq_len))

    def __getitem__(self, idx):
        clip = VideoFileClip(self.video_files[idx])
        if not clip.fps:
            clip.set_fps(25)
        if clip.end < (self.seq_len+1)/clip.fps:
            raise VideoTooShortException
        clip = clip.subclip(0, (self.seq_len+1)/clip.fps)
        frames = [resize(f, (200, 200)) for f in clip.iter_frames()]
        return np.array(frames[:-1]), np.array(frames[1:])

    def add_clip(self, filename):
        self.video_files.append(filename)

    def add_clips(self, file_list):
        self.video_files += file_list


if __name__ == "__main__":
    r = VideoReader()
    r.add_clip("training_data/1.webm")
    r.add_clips(["training_data/2.webm", "training_data/3.webm"])
    print(r.__len__())
    batch1_iputs, batch1_outputs = r.__getitem__(0)
    print(batch1_iputs.shape)
    print(batch1_outputs.shape)
    print(batch1_iputs)
