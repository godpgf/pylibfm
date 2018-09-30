# coding=utf-8
# author=godpgf
from ctypes import *
import numpy as np
from pyfm.libfm import fm
from .fm_matrix import FMMatrix
from .fm_vector import FMVector


class FM(object):
    def __init__(self, num_cols, id2group = None, num_factor=10, task="regression", algorithm = "sgd", learning_rate=0.001, init_stdev = 0.1, reg0 = 0.0, regw = 0.0, regv = 0.0, is_use_w0 = True, is_use_w = True):
        self.num_cols = num_cols
        self.num_group = num_cols if id2group is None else np.max(id2group) + 1
        self.id2group = id2group
        self.num_factor = num_factor
        self.task = task
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.init_stdev = init_stdev
        self.reg0 = reg0
        self.regw = regw
        self.regv = regv
        self.is_use_w0 = is_use_w0
        self.is_use_w = is_use_w
        self.p_fmModel = c_void_p(fm.createFMModel(c_int32(self.num_cols), c_int32(self.num_group), c_int32(self.num_factor), c_bool(self.is_use_w0), c_bool(self.is_use_w), c_float(self.init_stdev), c_float(self.reg0), c_float(self.regw), c_float(self.regv)))
        self.p_fmLearn = c_void_p(fm.createFM(c_char_p(self.task.encode()), c_char_p(self.algorithm.encode()), self.p_fmModel, c_float(self.learning_rate)))
        self.fm_x = FMMatrix(num_factor)
        self.fm_y = FMVector()

    def __del__(self):
        fm.releaseFMModel(self.p_fmModel)
        fm.releaseFM(self.p_fmLearn)

    def learn(self, x, y):
        if hasattr(x, 'indptr') and hasattr(x, 'indices') and hasattr(x, 'data'):
            self.fm_x.fill_sparse_matrix(x, self.id2group)
        elif isinstance(x, np):
            self.fm_x.fill_matrix(x)
        self.fm_y.fill(y)
        fm.learn(self.p_fmLearn, self.fm_x.p_sparse_matrix, self.fm_y.p_vector)

    def evaluate(self, x, y):
        if hasattr(x, 'indptr') and hasattr(x, 'indices') and hasattr(x, 'data'):
            self.fm_x.fill_sparse_matrix(x)
        elif isinstance(x, np):
            self.fm_x.fill_matrix(x)
        self.fm_y.fill(y)
        return fm.evaluate(self.p_fmLearn, self.fm_x.p_sparse_matrix, self.fm_y.p_vector)

    def predict(self, x):
        if hasattr(x, 'indptr') and hasattr(x, 'indices') and hasattr(x, 'data'):
            self.fm_x.fill_sparse_matrix(x)
            self.fm_y.fill_empty(len(x.indptr) - 1)
        elif isinstance(x, np):
            self.fm_x.fill_matrix(x)
            self.fm_y.fill_empty(len(x))

        fm.predict(self.p_fmLearn, self.fm_x.p_sparse_matrix, self.fm_y.p_vector)
        y = self.fm_y.to_array()
        return y
