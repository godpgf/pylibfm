# coding=utf-8
# author=godpgf
from ctypes import *
from pyfm.libfm import fm
from .fm_matrix import FMMatrix
from .fm_vector import FMVector


class FM(object):
    def __init__(self, num_factor=10, num_iter=100, task="regression", algorithm = "sgd", learning_rate=0.001, init_stdev = 0.1, reg0 = 0.0, regw = 0.0, regv = 0.0, is_use_w0 = True, is_use_w = True):
        self.num_factor = num_factor
        self.num_iter = num_iter
        self.task = task
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.init_stdev = init_stdev
        self.reg0 = reg0
        self.regw = regw
        self.regv = regv
        self.is_use_w0 = is_use_w0
        self.is_use_w = is_use_w
        self.p_fmModel = None
        self.p_fmLearn = None

    def fit(self, x, y):
        if self.p_fmModel:
            fm.releaseFMModel(self.p_fmModel)
        self.p_fmModel = c_void_p(fm.createFMModel(x.shape[1], self.num_factor, self.is_use_w0, self.is_use_w, c_double(self.init_stdev), c_double(self.reg0), c_double(self.regw), c_double(self.regv)))

        if self.p_fmLearn:
            fm.releaseFM(self.p_fmModel)
        self.p_fmLearn = c_void_p(fm.createFM(self.num_iter, c_char_p(self.task), c_char_p(self.algorithm), self.p_fmModel, c_double(self.learning_rate)))

        fm_x = FMMatrix(x)
        fm_y = FMVector(y)
        self._fit(fm_x, fm_y)
        fm_x.clean()
        fm_y.clean()

    def _fit(self, x, y):
        fm.learn(self.p_fmLearn, x.p_sparse_matrix, y.p_vector)

    def predict(self, x):
        fm_x = FMMatrix(x)
        fm_y = FMVector(dim=x.shape[0])
        self._predict(fm_x, fm_y)
        y = fm_y.to_array()
        fm_x.clean()
        fm_y.clean()
        return y

    def _predict(self, x, y):
        fm.predict(self.p_fmLearn, x.p_sparse_matrix, y.p_vector)