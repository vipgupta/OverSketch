# OverSketch

OverSketch.py function implements OverSketch from 
https://people.eecs.berkeley.edu/~vipul_gupta/oversketch.pdf

Running these files would require the pywren package (http://pywren.io/) and the numpywren package (https://github.com/Vaishaal/numpywren) to be installed.

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
