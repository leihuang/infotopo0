"""
A pathway of two reactions with simple mass action kinetics (mar2).

 __   R1     R2   __
|X1| <--> X <--> |X2| 
 --               --

X1, X2: constant concentrations

r1, r2: perturbation strength on k1 and k2
R1, R2: perturbation strength on X1 and X2
"""

from collections import OrderedDict as OD

import sympy
import numpy as np

from util.butil import Series

import predict
reload(predict)


k1, k2, X, X1, X2, r1, r2, R1, R2 = sympy.symbols('k1,k2,X,X1,X2,r1,r2,R1,R2')
v1 = r1*k1 * (R1*X1 - X)
v2 = r2*k2 * (X - R2*X2)

S = sympy.solve(v1-v2, X)[0]
J = sympy.simplify(v1.subs(X, S))

kwargs0 = {}

def _S(p, **kwargs):
    return np.float(S.evalf(subs={k1:p['k1'], k2:p['k2'], 
                                  X1:kwargs['X1'], X2:kwargs['X2'], 
                                  r1:kwargs['r1'], r2:kwargs['r2'], 
                                  R1:kwargs['R1'], R2:kwargs['R2']}))

def _J(p, **kwargs):
    return np.float(J.evalf(subs={k1:p['k1'], k2:p['k2'], 
                                  X1:kwargs['X1'], X2:kwargs['X2'], 
                                  r1:kwargs['r1'], r2:kwargs['r2'], 
                                  R1:kwargs['R1'], R2:kwargs['R2']}))

def get_predict_S(r1s=[1], r2s=[1], R1s=None, R2s=None):
    def predict_S(p):
        pass
    return predict_S


def get_predict_J(r1s=[1], r2s=[1], R1s=None, R2s=None):
    pass


pred_S = get_predict_S()
pred_J = get_predict_J()

pred = pred_S + pred_J




