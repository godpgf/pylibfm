# coding=utf-8
# author=godpgf
from ctypes import *
from pyfm.libfm import fm
import numpy as np


class FMVector(object):
    def __init__(self, vector = None, dim = None):
        if vector is not None:
            self.values = (c_float * len(vector))()
            for i in xrange(len(vector)):
                self.values[i] = vector[i]
            self.dim = len(vector)
        elif dim is not None:
            self.values = (c_float * dim)()
            self.dim = dim

        self.p_vector = c_void_p(fm.createDVector(self.values, self.dim))

    def to_array(self):
        fm.transformVector2Array(self.p_vector, self.values)
        return np.array([self.values[i] for i in xrange(self.dim)])

    def clean(self):
        fm.releaseDVector(self.p_vector);