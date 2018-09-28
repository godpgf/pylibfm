# coding=utf-8
# author=godpgf
from ctypes import *
from pyfm.libfm import fm
import numpy as np


class FMVector(object):
    def __init__(self):
        self.p_vector = c_void_p(fm.createDVector())
        self.max_num_values = 0
        self.data = None

    def __del__(self):
        fm.releaseDVector(self.p_vector);

    def to_array(self):
        fm.transformVector2Array(self.p_vector, self.data)
        return np.array([self.data[i] for i in range(self.dim)])

    def fill(self, values):
        if len(values) > self.max_num_values:
            self.max_num_values = len(values)
            self.data = (c_float * len(values))()
        for id, v in enumerate(values):
            self.data[id] = v
        self.dim = len(values)
        fm.fillDVector(self.p_vector, self.data, c_int32(self.dim))

    def fill_empty(self, size):
        if size > self.max_num_values:
            self.max_num_values = size
            self.data = (c_float * size)()
        self.dim = size
        fm.fillDVector(self.p_vector, self.data, c_int32(self.dim))