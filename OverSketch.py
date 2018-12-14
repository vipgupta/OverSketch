"""
This function implements OverSketch from 
https://people.eecs.berkeley.edu/~vipul_gupta/oversketch.pdf
It calculates A*B approximately using sketching on AWS Lambda.
Takes arguments: 
BigMatrix 'A' 
BigMatrix 'B' 
sketch dimension 'd', 
straggler threshold 'thres', default 0.95 

BigMatrix objects A (m x n) and B (n x l) should satisfy:
-A.shard_sizes[0] = B.shard_sizes[1] = b, where b is the block-size
-b should divide d, m and l
-Columns and rows of A and B, respectively, are unsharded in AWS S3 storage (however, this code can be trivially generalized)

Returns: A numpywren BigMatrix that contains the sketched product of A and B in S3
"""

import numpy as np
import pywren
import time
import numpywren
from numpywren import matrix
from numpywren import binops
from numpywren.matrix_init import shard_matrix


def OverSketchFunc(A, B, d, thres = 0.95):

    m = A.shape[0]
    n = A.shape[1]
    l = B.shape[1]
    b = A.shard_sizes[0]

    assert (d % b == 0)
    assert (m % b == 0)
    assert (l % b == 0)
    assert (b == B.shard_sizes[1])

    N = int(d/b)

    sketch_A = matrix.BigMatrix("sketch_A_{0}_{1}".format(m, d), shape=(m, d), shard_sizes=(b, b))
    sketch_BT = matrix.BigMatrix("sketch_B_{0}_{1}".format(l, d), shape=(l, d), shard_sizes=(b, b))

    hashes = np.random.randint(0, b, size=(N, n))
    flips = np.random.choice([-1,1], size=(N, n))

    def OverSketchMatrix(id, X, hashes, flips, b, sketch):
        """
        Calculates OverSketch AS for a row-block of a fat matrix A with block-size b
        """
        x = id[0]
        y = id[1]
        A = X.get_block(x,0)
        m,n = A.shape
        hash_local = hashes[y,:]
        flip_local = flips[y,:]
        sketch_block = np.zeros((m, b))
        for i in range(n):
            sketch_block[:, hash_local[i]] += flip_local[i]*A[:,i]
        sketch.put_block(sketch_block, x, y)
        return 0

    pwex = pywren.lambda_executor()

    t1 = time.time()
    futuresA = pwex.map(lambda x: OverSketchMatrix(x, A, hashes, flips, b, sketch_A), sketch_A.block_idxs)
    futuresB = pwex.map(lambda x: OverSketchMatrix(x, B.T, hashes, flips, b, sketch_BT), sketch_BT.block_idxs)
    fs_donesA = pywren.wait(futuresA, 2)[0]
    fs_donesB = pywren.wait(futuresB, 2)[0]
    while len(fs_donesA)<thres*len(futuresA) and len(fs_donesB)<thres*len(futuresB):
        fs_donesA = pywren.wait(futuresA, 2)[0]
        fs_donesB = pywren.wait(futuresB, 2)[0]
    print("Sketching time", time.time() - t1)

    ## Computation phase
    def blockMatMul(A, B, tensorAB, id):
        """
        Multiplies A and B.T in a blocked fashion
        """
        i = id[0]
        j = id[1]
        k = id[2]
        X = A.get_block(i,k)
        Y = B.get_block(j,k)
        tensorAB[k].put_block(X.dot(Y.T), i, j)
        return 0

    tensorAB = []
    for x in range(N):
        tensorAB.append(matrix.BigMatrix("AxB_outer_{0}_{1}_{2}".format(m, l, x), shape=(m, l), shard_sizes=(b, b)))

    computeArr = [(i,j,k) for (i,k) in sketch_A.block_idxs for j in sketch_BT._block_idxs(0)]

    t1 = time.time()
    futures = pwex.map(lambda x: blockMatMul(sketch_A, sketch_BT, tensorAB, x), computeArr)
    fs_dones = pywren.wait(futures, 2)[0]
    while len(fs_dones)<thres*len(futures):
        fs_dones = pywren.wait(futures, 2)[0]
    print("Computation time", time.time() - t1)

    ## Reduction phase

    def blockMatMulReduction(tensorAB, AB, id):
        """
        Reduces the output from computation phase to get A*B
        Variable 'count' keeps track of number of blocks that have returned 
        """
        i = id[0]
        j = id[1]
        X = None
        count = 1
        for k in range(N):
            if X is None:
                try:
                    X = tensorAB[k].get_block(i,j)
                except Exception as e:
                    print(e)
                    pass
            else:
                try:
                    X = X + tensorAB[k].get_block(i,j)
                    count = count+1
                except Exception as e:
                    print(e)
                    pass
        AB.put_block(X/count, i, j)  
        return 0

    AB = matrix.BigMatrix("AxB_{0}_{1}".format(m, l), shape=(m, l), shard_sizes=(b, b))
    reduceArr = [(i,j) for i in sketch_A._block_idxs(0) for j in sketch_BT._block_idxs(0)]

    t1 = time.time()
    futures_red = pwex.map(lambda x: blockMatMulReduction(tensorAB, AB, x), reduceArr)
    fs_dones = pywren.wait(futures_red)[0]
    print("Reduction time", time.time() - t1)

    return AB



