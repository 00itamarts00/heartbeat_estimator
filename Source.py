import numpy as np
import pandas as pd
import os, sys, re, datetime, glob
import matplotlib.pyplot as plt
import yaml, json, pickle
import cv2
import queue
from collections import deque
from Frame import Frame
import sys

class Source:
    def __init__(self, ecam_name, source_params):
        self.ecam = ecam_name
        self.stack = deque()
        self.params = source_params
        self.meta_path = os.path.join(self.params['recordings_path'], self.params['recordings_id'], self.ecam+'.json')
        self.recordings_path = os.path.join(self.params['recordings_path'], self.params['recordings_id'], self.ecam)
        self.stack_empty_check()

    def load_source_stack(self):
        meta_data = self.load_metadata_to_list()
        for meta in meta_data:
            self.stack.appendleft(meta)
        self.stack_empty_check()
        print(f'Done Stacking source of images from cam: {self.ecam}')

    def pop_frame_from_stack(self):
        if not self.stack_empty:
            return Frame(self.get_frame_img(self.stack.pop()), self.stack.pop())
        else:
            print(f' Source {self.ecam} is empty')
            sys.exit(' Source Ended')
        self.stack_empty_check()

    def get_frame_img(self, meta):
        # img_path = glob.glob(os.path.join(self.recordings_path,
        #                                   '{}*{}.png'.format(self.ecam, meta['$meta']['receive_time'])))[0]
        img_path = os.path.join(self.recordings_path,
                                '{}.0000000{}.png'.format(self.ecam, meta['$meta']['receive_time']))
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    def stack_empty_check(self):
        self.stack_empty = True if len(self.stack) == 0 else False

    def load_metadata_to_list(self):
        with open(self.meta_path) as f:
            content = f.readlines()
        content = [json.loads(x.strip()) for x in content]
        return content
