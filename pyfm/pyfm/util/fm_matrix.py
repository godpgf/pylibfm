# coding=utf-8
# author=godpgf
from ctypes import *
from pyfm.libfm import fm

#csr_matrix作为输入数据，设数组格式如下：
#[[1, 0, 2],
#[0, 0, 3],
#[4, 5, 6]]
#data 表示 元数据 显然为1， 2， 3， 4， 5， 6
#shape 表示 矩阵的形状 为 3 * 3
#indices 表示 各个数据在各行的下标， 从该数据我们可以知道：数据1在某行的0位置处， 数据2在某行的2位置处，6在某行的2位置处。
#indptr 表示每行数据的个数：[0 2 3 6]表示从第0行开始数据的个数，0表示默认起始点，0之后有几个数字就表示有几行，第一个数字2表示第一行有2 - 0 = 2个数字，因而数字1，2都第0行，第二行有3 - 2 = 1个数字，因而数字3在第1行，以此类推，我们能够知道所有数字的行号

class FMMatrix(object):
    def __init__(self, num_cols):
        self.p_sparse_matrix = c_void_p(fm.createSparseMatrix(c_int32(num_cols)))
        self.max_num_values = 0
        self.max_num_rows = 0
        self.num_cols = num_cols
        self.data = None
        self.indptr = None
        self.indices = None
        self.id2group = (c_int * num_cols)()

    def __del__(self):
        fm.releaseSparseMatrix(self.p_sparse_matrix)

    def fill_sparse_matrix(self, sx, id2group = None):
        if len(sx.indptr) - 1 > self.max_num_rows:
            self.max_num_rows = len(sx.indptr) - 1
            self.indptr = (c_int * len(sx.indptr))()

        if len(sx.data) > self.max_num_values:
            self.max_num_values = len(sx.data)
            self.data = (c_float * len(sx.data))()
            self.indices = (c_int * len(sx.data))()

        for id, v in enumerate(sx.indptr):
            self.indptr[id] = v
        for id, v in enumerate(sx.indices):
            self.indices[id] = v
        for id, v in enumerate(sx.data):
            self.data[id] = v
        if id2group is not None:
            for id, v in enumerate(id2group):
                self.id2group[id] = int(v)
        else:
            for id in range(self.num_cols):
                self.id2group[id] = id
        fm.fillSparseMatrix(self.p_sparse_matrix, self.data, self.indices, self.indptr, self.id2group, c_int32(len(sx.indptr)-1), c_int32(len(sx.data)))

    def fill_matrix(self, x):
        indptr = [0]
        data = []
        indices = []
        for line in x:
            for id, value in enumerate(line):
                if abs(value) > 0:
                    data.append(value)
                    indices.append(id)
                indptr.append(len(data))
        sx = object()
        sx.indptr = indptr
        sx.data = data
        sx.indices = indices
        self.fill_sparse_matrix(sx)
