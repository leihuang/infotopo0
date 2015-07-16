from sage.all import *

A = matrix(ZZ, [[-1.,1.],[1.,-1.]])

print A.kernel()