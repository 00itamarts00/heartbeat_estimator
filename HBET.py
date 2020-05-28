import numpy as np
import pandas as pd
import os, sys, re, datetime
import matplotlib.pyplot as plt
import yaml, json, pickle
import cv2
import queue
from collections import deque
from Buffer import Buffer
from Source import Source
from Frame import Frame
import tkinter
import padasip as pa
from FileHandler import load_yaml
from fft import get_pca_from_signals
import heapq


class HBET:
    def __init__(self):
        self.params = load_yaml('params_dict.yml')
        self.source = dict()
        self.buffer = dict()

    def init_sources(self):
        self.source = dict()
        source_params = self.params['source']
        for ecam_name in self.params['recording_cams']:
            self.source[ecam_name] = Source(ecam_name, source_params)

    def load_sources(self):
        for ecam_name in self.params['recording_cams']:
            self.source[ecam_name].load_source_stack()

    def init_buffers(self):
        buffer_params = self.params['buffer']
        for ecam_name in self.params['recording_cams']:
            self.buffer[ecam_name] = Buffer(ecam_name, buffer_params)

    def load_buffer_with_source_img(self, ecam_name):
        # TODO deal with short exposure frames
        item = self.source[ecam_name].pop_frame_from_stack()
        while item.short_exposure:
            item = self.source[ecam_name].pop_frame_from_stack()
        if self.params['buffer']['image_type'] == 'downscale':
            item.get_frame_pyramid_level(self.params['frame']['type'], self.params['frame']['level'])
        self.buffer[ecam_name].put_to_buffer(item)

    def load_buffer_until_full(self, ecam_name):
        while not self.buffer[ecam_name].full_buffer:
            self.load_buffer_with_source_img(ecam_name)
        print(f'Buffer {ecam_name} is full')

    def load_all_buffer_until_full(self):
        for ecam_name in self.params['recording_cams']:
            self.load_buffer_until_full(ecam_name)
        print('Done Loading all buffers until full')

    def update_all_buffers_indicators(self):
        for buffer in self.buffer.values():
            buffer.update_buffer_indicators()

    def buffer_single_step(self, ecam):
        self.buffer[ecam].get_from_buffer()
        self.load_buffer_with_source_img(ecam)

    def step_buffer_n_steps(self, ecam_name, n_step):
        for step in range(n_step):
            self.buffer_single_step(ecam_name)

    def step_all_buffers_n_steps(self, n_steps):
        for ecam_name in self.params['recording_cams']:
            self.step_buffer_n_steps(ecam_name, n_steps)

    def get_pca_from_signals(self, signals):
        pca_sgnl = np.transpose([buffer.buffer_key_array for buffer in self.buffer.values()])
        self.pca = pa.preprocess.pca.PCA(pca_sgnl, n=1)
        pass

    def get_mean_fps_buffer(self):
        return np.nanmean([buffer.fps for buffer in self.buffer.values()])


def main():
    hbet = HBET()
    hbet.init_sources()
    hbet.load_sources()
    hbet.init_buffers()
    hbet.load_all_buffer_until_full()
    hbet.update_all_buffers_indicators()
    for nstep in np.arange(0, 5000, 1):
        pca = get_pca_from_signals(
            np.asarray([buffer.buffer_key_array_fft_filtered for buffer in hbet.buffer.values()]))
        freq = np.mean([buffer.buffer_key_array_freq for buffer in hbet.buffer.values()], axis=0)
        cnt = 1
        bpm = freq[heapq.nlargest(cnt, range(len(pca)), pca.take)[-1]] * 60
        th = hbet.params['hbet']['valid_hb']
        while ((bpm <= th[0]) | (bpm >= th[1])):
            cnt += 1
            bpm = freq[heapq.nlargest(cnt, range(len(pca)), pca.take)[-1]] * 60
        print(f'BPM: {bpm}')
        hbet.step_all_buffers_n_steps(nstep)
        hbet.update_all_buffers_indicators()
    print('Done')


if __name__ == '__main__':
    main()
