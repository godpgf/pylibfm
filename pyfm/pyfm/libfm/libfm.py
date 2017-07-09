# coding=utf-8
# author=godpgf
import ctypes
from ctypes import c_void_p

def test_connect():
    fm = ctypes.cdll.LoadLibrary("../lib/libfm_api.so")
    from ctypes import c_char_p, c_double
    name = c_char_p("mystring")
    print fm.test_connect(1, 4, name, True, c_double(8.2))

#test_connect()
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
lib_path = curr_path + "/../../lib/"

fm = ctypes.cdll.LoadLibrary(lib_path + "libfm_api.so")
fm.createSparseMatrix.restype = c_void_p
fm.createDVector.restype = c_void_p
fm.createFM.restype = c_void_p
fm.createFMModel.restype = c_void_p