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


def load_yaml(param_dict):
    with open(param_dict, 'r') as stream:
        paramdict = yaml.safe_load(stream)
    return paramdict


def load_pickle_to_dict(pickle_path):
    return pickle.load(open(pickle_path, 'rb'))


def load_json_to_dict(json_path):
    with open(json_path, "r") as read_file:
        data = json.load(read_file)
    return data
