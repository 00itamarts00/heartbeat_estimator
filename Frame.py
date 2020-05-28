import numpy as np
import pandas as pd
import os, sys, re, datetime
import matplotlib.pyplot as plt
import yaml, json, pickle
import cv2
from skimage.transform import pyramid_gaussian
import queue
from collections import deque


class Frame:
    def __init__(self, raw_img, meta):
        self.raw_img = np.asarray(raw_img, dtype=float)
        self.meta = meta
        self.short_exposure = True if np.median(self.raw_img) < 50 else False
        self.timestamp = self.meta['$meta']['receive_time']

    def get_frame_pyramid_level(self, pyramid_type='laplacyian', level=3):
        self.pyramid_type = pyramid_type
        self.pyramid_level = level
        self.pyramid_img = []
        if pyramid_type == 'laplacyian':
            self.pyramid_img = self.get_laplacian_pyramid_layer(self.raw_img, level)
        elif pyramid_type == 'gaussian':
            self.pyramid_img = self.get_gaussian_pyramid(self.raw_img, level)

    @staticmethod
    def get_laplacian_pyramid_layer(gi, n):
        for i in range(n):
            gi_prev = gi
            gi = cv2.pyrDown(gi_prev)
        pyrup = cv2.pyrUp(gi)
        return cv2.addWeighted(gi_prev, 1.5, pyrup, -0.5, 0)

    @staticmethod
    def get_gaussian_pyramid(img, lvl):
        layer = img
        for i in range(lvl):
            layer = cv2.pyrDown(layer)
        return layer
