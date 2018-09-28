# coding=utf-8
# author=godpgf
import ctypes
import platform
from ctypes import c_void_p, c_float, c_bool, c_int32

sysstr = platform.system()

import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
lib_path = curr_path + "/../../lib/"

try:
    fm = ctypes.windll.LoadLibrary(lib_path + 'libfm_api.dll') if sysstr =="Windows" else ctypes.cdll.LoadLibrary(lib_path + 'libfm_api.so')
except OSError as e:
    lib_path = curr_path + "/../../../lib/"
    fm = ctypes.windll.LoadLibrary(
        lib_path + 'libfm_api.dll') if sysstr == "Windows" else ctypes.cdll.LoadLibrary(
        lib_path + 'libfm_api.so')


fm.createSparseMatrix.restype = c_void_p
fm.createDVector.restype = c_void_p
fm.createFM.restype = c_void_p
fm.createFMModel.restype = c_void_p
fm.evaluate.restype = c_float