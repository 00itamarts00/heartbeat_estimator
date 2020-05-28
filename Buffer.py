import numpy as np
import pandas as pd
import os, sys, re, datetime
import matplotlib.pyplot as plt
import yaml, json, pickle
import cv2
import queue
from collections import deque
from fft import butter_bandpass_filter, window_filter
from scipy.signal import medfilt


class Buffer:
    def __init__(self, ecam_name, buffer_params):
        self.buffer_name = ecam_name
        self.params = buffer_params
        self.max_buffer_size = self.params['max_size']
        self.queue = queue.Queue(maxsize=self.max_buffer_size)
        self.update_buffer_size_status()

    def get_from_buffer(self):
        tmp = self.queue.get()
        self.update_buffer_size_status()
        return tmp

    def put_to_buffer(self, item):
        if not self.full_buffer:
            self.queue.put(item)
        else:
            print(f'Buffer is full, when putting item {item}')
        self.update_buffer_size_status()

    def update_buffer_size_status(self):
        self.buffer_size = self.queue.qsize()
        self.full_buffer = True if self.buffer_size == self.max_buffer_size else False

    def get_imgs_cube(self):
        mtx = []
        for elem in list(self.queue.queue):
            mtx.append(elem.pyramid_img)
        return np.diff(mtx, axis=0)

    def get_buffer_std_img(self):
        mtx = self.get_imgs_cube()
        self.std_img = np.asarray(np.nanstd(mtx, axis=0), dtype=float)

    def get_buffer_std_mask(self):
        self.get_buffer_std_img()
        mask_th = self.params['std_mask_limits']
        mask = np.ones_like(self.std_img, dtype=bool)
        mask[(self.std_img > mask_th[1]) | (self.std_img <= mask_th[0])] = False
        self.std_mask = mask

    def get_frame_key_value(self, elem):
        # Removing outliers
        arr = elem.pyramid_img[self.std_mask]
        arr = np.delete(arr, np.where(arr <= self.params['minimum_gl_th']))
        mean, std = np.nanstd(arr), np.nanmean(arr)
        arr_filtered = arr[(arr <= mean + 3*std**0.5) & (arr >= mean - 3*std**0.5)]
        return np.nanmean(arr_filtered)

    def get_buffer_key_array(self):
        buffer_key_array = []
        for elem in list(self.queue.queue):
            elem.key_val = self.get_frame_key_value(elem)
            buffer_key_array.append(elem.key_val)
        self.buffer_key_array = buffer_key_array

    #TODO
    def get_buffer_key_array_fft_filtered(self):
        filter_type = self.params['fft']['type']
        filter_threshold = self.params['fft']['threshold']
        if filter_type == 'butter':
            self.buffer_key_array_fft_filtered = butter_bandpass_filter(self.buffer_key_array, filter_threshold[0],
                                                                        filter_threshold[1], self.fps)
        elif filter_type == 'window':
            self.buffer_key_array_freq, self.buffer_key_array_fft_filtered = \
                window_filter(self.buffer_key_array, filter_threshold[0],
                              filter_threshold[1], self.fps, return_time_domain=False)
        self.buffer_key_array_fft_filtered *= self.params['fft']['amplification_factor']

    def calc_fps_of_buffer(self):
        time_arr = []
        for elem in self.queue.queue:
            time_arr.append(elem.timestamp)
        self.fps  = 1/((np.abs(np.mean(np.diff(time_arr))))/1e9)

    def update_buffer_indicators(self):
        self.get_buffer_std_mask()
        self.get_buffer_key_array()
        self.calc_fps_of_buffer()
        self.get_buffer_key_array_fft_filtered()
