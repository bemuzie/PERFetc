# -*- coding: utf-8-*-
# cython: profile=False
# Copyright 2012 Denis Nesterov
#   cireto@gmail.com
#
#   The software is licenced under BSD licence.
"""
  A cython implementation of bilateral filtering.
"""

import numpy as np

cimport numpy as np
cimport cython

from cython cimport view

from libc.stdlib cimport abs as c_abs

DTYPE = np.int
DTYPEfloat = np.float64
ctypedef np.int_t DTYPE_t
ctypedef np.float32_t DTYPEfloat_t



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
#sme as data_2 but different looping/
def ssd(double [:,::1] data, double[:, ::1] curves):
    cdef int d_row_num = data.shape[0]
    cdef int c_row_num = curves.shape[0]
    cdef int col_num = curves.shape[1]
    cdef int data_row, curve_row, col
    cdef int diff_min_idx
    cdef double diff, diff_new, diff_col
    cdef long[::1] out_array = np.zeros(d_row_num, dtype=DTYPE)
    cdef double[::1] ssds = np.zeros(d_row_num, dtype=DTYPEfloat)
    cdef double[:] data_subset
    cdef double[:] curve_subset

    diff_min_idx = 0
    for data_row in range(d_row_num):
        diff = 0
        for col in range(col_num):
            diff_col = data[data_row, col] - curves[diff_min_idx, col]
            diff_col *= diff_col
            diff += diff_col

        if data_row % 50000 == 0:
            print data_row, d_row_num

        for curve_row in range(0, c_row_num):
            diff_new = 0
            for col in range(col_num):
                diff_col = data[data_row, col] - curves[curve_row, col]
                diff_col *= diff_col
                diff_new += diff_col
                if diff_new > diff:
                    break
            if diff_new < diff:
                diff = diff_new
                diff_min_idx = curve_row

        out_array[data_row] = diff_min_idx
        ssds[data_row] = diff
    return out_array, ssds






