"""
Test file for checking OverSketch
"""

import numpy as np
import pywren
import time
import numpywren
from numpywren import matrix, matrix_utils
from numpywren import binops
from numpywren.matrix_init import shard_matrix
from OverSketch import OverSketchFunc


m = 2000
n = 10000
b = 1000
l = 3000
d = int(4*b)

A_loc = np.asarray(range(m*n))
A_loc = A_loc.reshape(m,n)
B_loc = np.random.rand(n,l)
A = matrix.BigMatrix("oversketch_A_{0}_{1}_{2}".format(m, n, b), shape=(m, n), shard_sizes=(b, n), write_header=True)
shard_matrix(A, A_loc)
B = matrix.BigMatrix("oversketch_B_{0}_{1}_{2}".format(n, l, b), shape=(n, l), shard_sizes=(n, b), write_header=True)
shard_matrix(B, B_loc)

print("A and B done")

AB = OverSketchFunc(A, B, d)

print("OverSketch done")

c = AB.numpy()
ab_exact = A_loc.dot(B_loc)
error = np.linalg.norm(ab_exact-c, ord='fro')/np.linalg.norm(ab_exact,ord='fro')

print("Frobenius norm error in OverSketched product:{:.2f}%".format(error*100))