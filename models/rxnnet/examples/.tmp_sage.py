from sage.all import *

A = matrix(RR, [[-1.,0.],[1.,-1.],[0.,1.]])

print A.kernel()