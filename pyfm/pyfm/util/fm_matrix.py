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
    def __init__(self, sx):
        dim = sx.shape[0]
        self.p_sparse_matrix = c_void_p(fm.createSparseMatrix(dim))
        line_id = 0
        for i in xrange(len(sx.indptr)-1):
            line = list()
            for j in xrange(sx.indptr[i], sx.indptr[i+1]):
                line.append((sx.indices[j],sx.data[j]))
            self.fill_line(line_id, line)
            line_id += 1

    def fill_line(self, line_id, data):
        ids =  (c_int * len(data))()
        values = (c_float * len(data))()
        for i in xrange(len(data)):
            ids[i] = data[i][0]
            values[i] = data[i][1]
        fm.fillSparseMatrixLine(self.p_sparse_matrix, line_id, ids, values, len(data))

    def clean(self):
        fm.releaseSparseMatrix(self.p_sparse_matrix)
